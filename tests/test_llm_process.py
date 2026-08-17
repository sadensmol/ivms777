import sys

import pytest

from config import Settings
from modelsvc.llm_process import RemoteLlm, SubprocessLlm, build_llm_process


def test_subprocess_loads_waits_healthy_and_frees():
    calls = {"n": 0}

    def probe(url):  # healthy on the 2nd poll
        calls["n"] += 1
        return calls["n"] >= 2

    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    llm = SubprocessLlm(
        cmd, health_url="http://x/health", ready_timeout_s=5, probe=probe, poll_interval_s=0.01
    )
    llm.load()
    assert llm.is_loaded()
    assert calls["n"] >= 2
    llm.free()
    assert not llm.is_loaded()


def test_subprocess_raises_and_cleans_up_on_unhealthy():
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    llm = SubprocessLlm(
        cmd,
        health_url="http://x/health",
        ready_timeout_s=0.05,
        probe=lambda url: False,
        poll_interval_s=0.01,
    )
    with pytest.raises(TimeoutError):
        llm.load()
    assert not llm.is_loaded()  # child was killed


def test_subprocess_switching_mode_restarts_the_child():
    # One child, one port: text and vision are mutually exclusive, so asking for the
    # other mode must RESTART the process with the new command (design §3.1).
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    llm = SubprocessLlm(
        cmd, health_url="http://x/health", vision_args=["--mmproj", "/m.gguf"],
        ready_timeout_s=5, probe=lambda url: True, poll_interval_s=0.01,
    )
    llm.load(vision=False)
    text_pid = llm._proc.pid
    llm.load(vision=False)
    assert llm._proc.pid == text_pid          # same mode -> no restart

    llm.load(vision=True)
    assert llm._proc.pid != text_pid          # mode changed -> restarted
    assert "--mmproj" in llm._proc.args
    llm.free()


def test_subprocess_load_is_idempotent_while_alive():
    calls = {"n": 0}

    def probe(url):
        calls["n"] += 1
        return True

    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    llm = SubprocessLlm(cmd, health_url="http://x/health", probe=probe, poll_interval_s=0.01)
    llm.load()
    first = llm._proc
    llm.load()  # already alive -> no respawn
    assert llm._proc is first
    llm.free()


def test_subprocess_reaps_a_child_that_exits_early():
    # A child that dies at startup (e.g. gemma OOM at model load) must be reaped, not
    # left as a <defunct> zombie — load() raises AND clears the handle.
    cmd = [sys.executable, "-c", "raise SystemExit(1)"]
    llm = SubprocessLlm(
        cmd, health_url="http://x/health", ready_timeout_s=5,
        probe=lambda url: False, poll_interval_s=0.01,
    )
    with pytest.raises(RuntimeError):
        llm.load()
    assert llm._proc is None       # reaped, no lingering zombie
    assert not llm.is_loaded()


def test_build_llm_process_pins_all_layers_on_gpu():
    # GPU-ONLY, ALWAYS (design §3.1/§8.1): jetson pins every layer on the GPU. There
    # is no CPU-offload fallback — gemma is made to fit (Q8 projector, small KV), and
    # a load that does not fit aborts loudly instead of silently running on the CPU.
    llm = build_llm_process(Settings(profile="jetson"))
    assert isinstance(llm, SubprocessLlm)
    assert llm._cmd[llm._cmd.index("-ngl") + 1] == "99"


def test_projector_loads_only_for_vision_mode():
    # The projector is gemma's VISION half — captioning only. A text chat must never
    # pay its ~531 MB, so it is absent from the text command and present in the
    # vision one (design §3.1).
    llm = build_llm_process(Settings(profile="jetson"))
    text_cmd = llm.command_for(vision=False)
    vision_cmd = llm.command_for(vision=True)
    assert not any("mmproj" in a for a in text_cmd)
    assert "--mmproj" in vision_cmd
    assert any("mmproj-gemma-4-E2B-it-Q8_0.gguf" in a for a in vision_cmd)
    # Vision is the text command PLUS the projector — same weights, same flags.
    assert vision_cmd[: len(text_cmd)] == text_cmd


def test_build_llm_process_pins_ngl_when_set():
    llm = build_llm_process(Settings(profile="jetson", llm_ngl=20))
    assert llm._cmd[llm._cmd.index("-ngl") + 1] == "20"


def test_build_llm_process_uses_profile_ctx():
    # jetson shrinks the KV cache to 2048 to fit the 8 GB box; mac keeps 4096.
    assert build_llm_process(Settings(profile="jetson"))._cmd[
        build_llm_process(Settings(profile="jetson"))._cmd.index("-c") + 1
    ] == "2048"


def test_build_llm_process_mmproj_name_is_configurable():
    # Swap the projector without a code change (IVMS777_MMPROJ_NAME).
    llm = build_llm_process(Settings(profile="jetson", mmproj_name="mmproj-F16.gguf"))
    vision_cmd = llm.command_for(vision=True)
    assert any("mmproj-F16.gguf" in a for a in vision_cmd)
    assert not any("Q8_0" in a for a in vision_cmd)


def test_remote_llm_lifecycle_is_noop():
    llm = RemoteLlm(health_url="http://x/health", probe=lambda url: True)
    llm.load()
    llm.free()  # no raise, no process
    assert llm.is_loaded() is True
