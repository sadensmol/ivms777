"""The effect library (design §11.1) — the vocabulary a memory film is cut in.

Every effect is a **function of the image plus four numbers**: a focus point, an
amount, a speed, a direction. The numbers come from the model (`feed/conductor.py`),
which is looking at what the photograph contains and where in the music it lands. This
module's job is not taste — it is to say what is *possible*, to describe each effect
well enough that a small model can choose the right one, and to repair whatever comes
back.

PURE data plus validation. No model, no I/O, no torch (CLAUDE.md § one model process).

**Every entry carries `feel`, `use_when` and `avoid_when`, and every parameter carries
its own note.** That is not documentation for us — it is the prompt. A model handed
`{"bloom": "id 2"}` picks at random; a model told *"highlights swell and bleed like
light through a dirty lens; reach for it on snow, low sun and street lamps; never on a
flat grey interior, where it just washes the picture out"* picks like an editor. The
description IS the interface, so it is written for a reader who cannot see the photo
twice.

**Parameters pack positionally into a `vec4`.** `PARAM_ORDER[name]` is the contract:
slot 0 of `bloom` is its threshold, and the shader hardcodes nothing else. Four is not
a limit to work around — an effect wanting a fifth number is two effects.

A name or a number the model invents is REPAIRED, never fatal: an unknown effect falls
back to the director's own choice, an out-of-range number is clamped. The worst a
confused model can do is make a shot look ordinary.
"""

# Each param is `(low, high, default, note)`. The dict ORDER is the vec4 layout.
SHOT_EFFECTS: dict[str, dict] = {
    "plain": {
        "id": 0,
        "energy": ("high", "calm", "low"),
        "feel": "Nothing over the picture at all — just the camera's slow push. Clean, "
                "quiet, and completely out of the way.",
        "use_when": "The photograph is already doing the work: a strong face, a busy "
                    "street, anything you want read rather than decorated. Also the "
                    "right answer after two or three treated shots in a row, to let "
                    "the film breathe.",
        "avoid_when": "Never wrong, but three plains in a row is a slideshow again.",
        "params": {},
    },
    "breeze": {
        "id": 6,
        "energy": ("calm", "low"),
        "feel": "The photograph floats. A slow, soft, wandering warp — a few pixels of "
                "sway, like a print hanging in still air or heat coming off a road. "
                "You feel it more than you see it.",
        "use_when": "Quiet, warm, unhurried moments: an afternoon, a garden, water, "
                    "curtains, a long slow passage of the score. It is the gentlest "
                    "thing in this library and the safest way to make a still photo "
                    "feel alive.",
        "avoid_when": "Sharp architecture or text — straight lines wobbling reads as a "
                      "broken screen, not as air.",
        "params": {
            "amount": (0.0, 1.0, 0.45, ("How far it sways. 0.2 is barely there, 1.0 is "
                                       "underwater. Stay under 0.5 for realism.")),
            "speed":  (0.1, 2.0, 0.6, ("Cycles per second. Under 0.5 is a long slow "
                                      "breath; over 1.2 starts to look nervous.")),
            "scale":  (0.2, 3.0, 1.2, ("Size of the wave. Low = one big slow bend of "
                                      "the whole frame, high = many small ripples.")),
            "warp":   (0.0, 1.0, 0.3, ("How much the sway twists rather than slides. 0 "
                                      "is a pure drift, 1 is a shimmer.")),
        },
    },
    "rack_focus": {
        "id": 1,
        "energy": ("high", "calm", "low"),
        "feel": "The shot opens out of focus and resolves onto one point, the way an "
                "eye lands on something. Soft, then suddenly clear.",
        "use_when": "There is exactly ONE subject and you can say where it is — a face, "
                    "a sign, a hand, a lit window. Strongest on the shot the film is "
                    "actually about, and on the first shot after a chapter break.",
        "avoid_when": "Wide landscapes and crowds — there is nothing for the eye to "
                      "arrive at, so it just looks like a blurry photo.",
        "params": {
            "x":      (0.0, 1.0, 0.5, ("Focus point across the frame. 0 is the left "
                                      "edge, 1 the right. Put it ON the subject.")),
            "y":      (0.0, 1.0, 0.5, ("Focus point down the frame. 0 is the TOP, 1 the "
                                      "bottom. Faces usually sit near 0.35.")),
            "amount": (0.0, 1.0, 0.6, ("How soft it starts. 0.3 is a gentle settle, 0.9 "
                                      "is unreadable for a moment — dramatic, use once.")),
            "settle": (0.15, 1.0, 0.45, ("Fraction of the shot spent resolving. 0.2 "
                                        "snaps into focus, 0.8 is a long slow arrival.")),
        },
    },
    "bloom": {
        "id": 2,
        "energy": ("calm", "high"),
        "feel": "Highlights swell and bleed, like light through a dirty lens. Snow "
                "glows, lamps flare, skin goes warm. Romantic and slightly nostalgic.",
        "use_when": "Backlit, sunlit, snowy, candlelit, neon — anything with bright "
                    "areas worth spilling. It is the single best effect for making an "
                    "ordinary daylight photo feel remembered rather than recorded.",
        "avoid_when": "Flat grey interiors and overcast days: there are no highlights "
                      "to bloom, so it only washes the picture out.",
        "params": {
            "threshold": (0.3, 0.95, 0.62, ("How bright a pixel must be to glow. LOW "
                                           "(0.35) makes the whole frame hazy; high "
                                           "(0.85) touches only true highlights.")),
            "strength":  (0.0, 1.0, 0.5, ("How much glow is added back. Past 0.7 the "
                                         "picture starts to lose its detail.")),
            "warmth":    (-1.0, 1.0, 0.35, ("Colour of the glow. Positive is golden "
                                           "(sun, lamps, skin), negative is cold blue "
                                           "(snow, dusk, screens).")),
            "radius":    (0.2, 1.0, 0.5, ("How far the light spreads. Small is a crisp "
                                         "sparkle, large is a soft wash.")),
        },
    },
    "weave": {
        "id": 3,
        "energy": ("high", "calm", "low"),
        "feel": "The frame breathes and drifts by a hair, with faint colour fringing at "
                "the edges — the picture is being projected, not displayed. Subtle, "
                "constant, and it makes everything look shot on film.",
        "use_when": "Almost any ordinary shot you want to feel cinematic rather than "
                    "digital. It is the workhorse: undramatic, and it never fights the "
                    "photograph.",
        "avoid_when": "Alongside `pulse` or `glitch` in the same shot — two kinds of "
                      "instability at once just reads as a fault.",
        "params": {
            "weave":   (0.0, 1.0, 0.5, ("How far the frame drifts. Keep under 0.7; "
                                       "beyond that it stops being a projector and "
                                       "starts being a shake.")),
            "fringe":  (0.0, 1.0, 0.45, ("Red/blue separation at the edges. 0.2 is a "
                                       "real lens, 0.8 is a broken one.")),
            "speed":   (0.2, 2.0, 1.0, "How fast it wanders. 1.0 is a projector gate."),
            "flicker": (0.0, 1.0, 0.25, ("Exposure instability, like an old bulb. A "
                                        "little is warmth; a lot is a strobe.")),
        },
    },
    "bokeh": {
        "id": 4,
        "energy": ("calm", "low"),
        "feel": "Soft round motes of light drift across the frame, in front of the "
                "picture and out of focus. Night, warmth, celebration.",
        "use_when": "Low light with points of light in it: streetlamps, candles, fairy "
                    "lights, rain on glass, a city at night, snow falling.",
        "avoid_when": "Bright daylight — floating discs in clear noon sun read as dirt "
                      "on the screen.",
        "params": {
            "density": (0.0, 1.0, 0.45, ("How many motes. Above 0.7 they compete with "
                                        "the photograph instead of framing it.")),
            "size":    (0.1, 1.0, 0.4, ("How large each one is. Big and few is elegant; "
                                       "small and many is snow.")),
            "drift":   (0.0, 1.0, 0.35, ("How fast they travel. 0.1 is nearly still, "
                                        "0.8 is wind.")),
            "warmth":  (-1.0, 1.0, 0.2, ("Positive is candle and sodium light, negative "
                                        "is moonlight and screens.")),
        },
    },
    "pulse": {
        "id": 5,
        "energy": ("high", "calm"),
        "feel": "Exposure and vignette breathe ON the beat — the picture opening a "
                "fraction brighter every downbeat. You see the music.",
        "use_when": "The score has an audible pulse and the moment has energy: a party, "
                    "a walk, a crowd, a montage. Put it where you want the film to feel "
                    "like it is moving forward.",
        "avoid_when": "Still, quiet, sad moments — a throbbing picture over a slow "
                      "passage is the most obvious mistake in this library.",
        "params": {
            "bpm":      (30.0, 180.0, 72.0, ("Beats per minute to breathe on. USE THE "
                                            "TEMPO GIVEN IN THE BRIEF, or the picture "
                                            "pulses against the music instead of with "
                                            "it.")),
            "depth":    (0.0, 1.0, 0.35, ("How much the exposure moves. 0.2 is a "
                                         "heartbeat, 0.8 is a nightclub.")),
            "phase":    (0.0, 1.0, 0.0, ("Offset onto the downbeat, in fractions of a "
                                        "beat. Leave at 0 unless the shot starts "
                                        "off-beat.")),
            "vignette": (0.0, 1.0, 0.4, "How much the edges close in with each pulse."),
        },
    },
    "glitch": {
        "id": 7,
        "energy": ("high",),
        "feel": "Quick, strong, ugly-on-purpose: the image tears into blocks, channels "
                "slip sideways, pixels coarsen for a fraction of a second and then it "
                "is clean again. Digital, aggressive, modern.",
        "use_when": "A hard accent — the exact frame a beat lands on, a hard cut into a "
                    "new chapter, or a burst of retakes you want to feel frantic. Its "
                    "power is entirely in being RARE: once or twice in a whole film.",
        "avoid_when": "Tender, slow or sad memories, and anywhere near a face you want "
                      "the viewer to love. It is a punctuation mark, never a mood.",
        "params": {
            "amount":  (0.0, 1.0, 0.5, ("How violent the tear is. 0.3 is a hiccup, 1.0 "
                                       "is a broken signal.")),
            "blocks":  (0.0, 1.0, 0.45, ("How coarse the pixelation gets at the peak. "
                                        "High values turn the photo into large flat "
                                        "squares.")),
            "rate":    (0.5, 8.0, 3.0, ("Bursts per second. Low is one deliberate stab, "
                                       "high is a stutter.")),
            "duration": (0.05, 1.0, 0.18, ("Fraction of the shot that glitches, measured "
                                          "from its START. Keep it short — the effect "
                                          "is an accent, not a state.")),
        },
    },
    "light_leak": {
        "id": 8,
        "energy": ("high", "calm", "low"),
        "feel": "A band of warm light sweeps across the frame and out, like the back of "
                "an old camera opening for a moment. Analogue, accidental, nostalgic.",
        "use_when": "Transitions between parts of a day, the last shot of a chapter, or "
                    "any memory you want to feel found rather than filed. Beautiful "
                    "over sunsets and interiors.",
        "avoid_when": "More than once every few shots — it stops being an accident and "
                      "starts being a filter.",
        "params": {
            "angle":    (0.0, 1.0, 0.25, ("Direction of the sweep, as a fraction of a "
                                         "full turn. 0 travels left to right, 0.25 "
                                         "bottom to top.")),
            "strength": (0.0, 1.0, 0.5, "How opaque the band is at its centre."),
            "width":    (0.05, 1.0, 0.3, ("How wide the band is. Narrow is a streak, "
                                         "wide is a wash over the whole frame.")),
            "warmth":   (-1.0, 1.0, 0.6, ("Positive is orange and red (sun, film), "
                                         "negative is cyan (fluorescent, dusk).")),
        },
    },
    # --- optics: what the LENS does with light (ids 9-13) -------------------------
    "anamorphic_flare": {
        "id": 9,
        "energy": ("high", "calm"),
        "feel": "Every bright point grows a long horizontal streak of cold blue light, "
                "the way a cinema lens smears a headlamp sideways across the whole "
                "frame. Widescreen, expensive, slightly science-fiction.",
        "use_when": "Points of light against dark: neon, headlights, a window at night, "
                    "sun through trees, a stage, a phone screen in a dark room. It is "
                    "what makes a snapshot look like it was shot on a cinema camera.",
        "avoid_when": "Flat daylight and pale interiors — with no point of light to "
                      "catch it only smears the picture sideways and softens faces.",
        "params": {
            "threshold": (0.3, 0.95, 0.7, ("How bright a pixel must be to streak. LOW "
                                           "(0.4) streaks half the picture into a blur; "
                                           "high (0.9) catches only true lamps.")),
            "strength":  (0.0, 1.0, 0.45, ("How bright the streak is. 0.25 is a hint of "
                                           "a lens, 0.9 is a full sci-fi flare.")),
            "hue":       (0.0, 1.0, 0.15, ("Colour of the streak. 0 is the cold blue of "
                                           "an anamorphic lens, 1 is warm gold for "
                                           "candles, sunset and sodium lamps.")),
            "length":    (0.1, 1.0, 0.55, ("How far it reaches across the frame. Short "
                                           "is a glint, long crosses the whole picture "
                                           "and costs the most to draw.")),
        },
    },
    "god_rays": {
        "id": 10,
        "energy": ("calm", "low", "high"),
        "feel": "Shafts of light fan out from one bright place in the picture and reach "
                "across the rest of it, the way sun comes through a window into dust or "
                "through cloud onto water. Still, wide, slightly holy.",
        "use_when": "There is ONE source of light and you can point at it: a window, a "
                    "gap in trees, the sun low over water, a doorway, a lamp in a dark "
                    "room. Strongest on the shot a chapter opens on.",
        "avoid_when": "Evenly lit pictures and flat overcast light — with no single "
                      "source the shafts come from nowhere and only haze the frame.",
        "params": {
            "x":        (0.0, 1.0, 0.5, ("Where the light comes from, across the frame. "
                                         "0 is the left edge, 1 the right. Put it ON "
                                         "the window or the sun.")),
            "y":        (0.0, 1.0, 0.3, ("Where the light comes from, down the frame. 0 "
                                         "is the TOP, 1 the bottom. Sun and windows sit "
                                         "near 0.2.")),
            "strength": (0.0, 1.0, 0.5, ("How solid the shafts are. 0.25 is air and "
                                         "dust, 0.9 is a beam you could stand in.")),
            "bpm":      (0.0, 180.0, 0.0, ("Leave at 0 and the shafts hold perfectly "
                                           "steady. Set it to the TEMPO GIVEN IN THE "
                                           "BRIEF and they swell on every downbeat "
                                           "instead — energy, not stillness.")),
        },
    },
    "prism": {
        "id": 11,
        "energy": ("high", "calm"),
        "feel": "The middle of the picture is clean and the edges break into colour — "
                "red pulling one way, blue the other — like a cheap lens or a piece of "
                "glass held in front of a good one. Dreamy low, psychedelic high.",
        "use_when": "The subject is in the CENTRE and the edges are ordinary: a "
                    "portrait, a single object, a face in a crowd. Also good on "
                    "anything that should feel remembered rather than seen.",
        "avoid_when": "Text, straight architecture, and any picture whose subject sits "
                      "near an edge — that is exactly where the colour breaks, and a "
                      "split face reads as a broken screen. Never with `weave`, which "
                      "already fringes.",
        "params": {
            "amount":  (0.0, 1.0, 0.45, ("How far red and blue separate at the corners. "
                                         "0.2 is a real lens, 1.0 is a rainbow.")),
            "clean":   (0.0, 0.9, 0.35, ("How much of the middle stays untouched. 0 "
                                         "splits from the very centre, 0.8 leaves the "
                                         "subject clean and bends only the corners.")),
            "twist":   (0.0, 1.0, 0.0, ("Direction of the split. 0 is straight out from "
                                        "the centre like glass, 1 is sideways around "
                                        "it like a swirl.")),
            "breathe": (0.0, 1.0, 0.3, ("How much the split grows and shrinks on its "
                                        "own. 0 holds it steady, 1 pulses slowly.")),
        },
    },
    "soft_focus": {
        "id": 12,
        "energy": ("low", "calm"),
        "feel": "The whole picture goes gentle — edges lose their bite and a soft light "
                "fills the SHADOWS rather than spilling off the highlights. Skin is "
                "flattered, contrast drops, everything reads a decade older.",
        "use_when": "Faces above everything else: a portrait, a child, someone asleep, "
                    "a couple. The right answer for a tender or grieving passage, and "
                    "for old scanned prints. Unlike `bloom` it needs no bright areas at "
                    "all — it works on a grey day and in a dim room.",
        "avoid_when": "Anything the viewer needs to READ or count: text, a landscape "
                      "with detail worth keeping, a busy street. It removes detail on "
                      "purpose.",
        "params": {
            "softness": (0.0, 1.0, 0.5, ("How much sharpness is given up. 0.25 is a "
                                         "flattering lens, 0.9 is a memory you cannot "
                                         "quite bring into focus.")),
            "glow":     (0.0, 1.0, 0.45, ("How much light bleeds into the DARK parts. "
                                          "This is the half that makes it a portrait "
                                          "rather than a blur; past 0.8 shadows go "
                                          "milky.")),
            "radius":   (0.0, 1.0, 0.4, ("How far the softness spreads. Small keeps the "
                                         "shape of a face, large dissolves the frame.")),
            "warmth":   (-1.0, 1.0, 0.3, ("Colour of the veil. Positive is gold and "
                                          "skin, negative is cold blue for dusk, snow "
                                          "and sadness.")),
        },
    },
    "spotlight": {
        "id": 13,
        "energy": ("low", "calm", "high"),
        "feel": "The whole frame falls away into darkness except one soft oval of "
                "light, and that oval drifts slowly as the shot runs. Theatrical, "
                "private, and it tells the viewer exactly where to look.",
        "use_when": "One thing in a busy picture matters and you can say where it is — "
                    "a face in a group, a hand, an object on a table, a figure in a "
                    "wide room. Very strong on the last shot before a dip to black.",
        "avoid_when": "Landscapes and anything whose subject fills the frame — there is "
                      "nothing to exclude, so it only darkens the corners twice, once "
                      "here and once with the film's own vignette.",
        "params": {
            "x":    (0.0, 1.0, 0.5, ("Centre of the light across the frame. 0 is the "
                                     "left edge, 1 the right. Put it ON the subject.")),
            "y":    (0.0, 1.0, 0.45, ("Centre of the light down the frame. 0 is the "
                                      "TOP, 1 the bottom. Faces sit near 0.35.")),
            "size": (0.08, 0.8, 0.32, ("How wide the lit oval is, as a fraction of the "
                                       "frame. 0.12 picks out one face, 0.6 is a gentle "
                                       "glow over half the picture.")),
            "dim":  (0.0, 1.0, 0.6, ("How dark everything outside goes. 0.3 is a soft "
                                     "closing-in, 1.0 leaves the rest almost black.")),
        },
    },
    # --- colour: the same picture, developed differently (ids 14-19) ---------------
    "duotone": {
        "id": 14,
        "energy": ("high", "low"),
        "feel": "The photograph is reprinted in two inks — one colour through the "
                "shadows, another through the lights, and nothing else. Graphic and "
                "poster-like: every shape survives, every real colour is thrown away.",
        "use_when": "Strong shapes and weak colour: silhouettes against a sky, a crowd, "
                    "a street at night, anything already close to monochrome. The "
                    "effect for a title shot or the frame a chapter opens on.",
        "avoid_when": "Faces you want recognised — skin in the wrong ink reads as "
                      "illness — and any shot whose whole point IS its colour: autumn, "
                      "a sunset, a red dress. Never in the same shot as another colour "
                      "effect; two grades at once is mud.",
        "params": {
            "shadow": (0.0, 1.0, 0.62, ("Hue of the dark ink, as a fraction round the "
                                        "wheel: 0 red, 0.1 amber, 0.33 green, 0.62 "
                                        "blue, 0.85 magenta. Cold shadows read as "
                                        "night, warm ones as sepia.")),
            "light":  (0.0, 1.0, 0.11, ("Hue of the bright ink, same wheel. Put it FAR "
                                        "from shadow — two neighbouring hues make mud. "
                                        "Blue shadows into amber lights is the classic "
                                        "pair.")),
            "amount": (0.0, 1.0, 0.42, ("How far the inks replace the real colour. 0.3 "
                                        "is a tinted photograph, 0.7 is a poster. Past "
                                        "0.8 it stops reading as the same picture.")),
            "ink":    (0.0, 1.0, 0.55, ("How saturated the two inks are. 0 is plain "
                                        "black-and-white, 1 is silkscreen.")),
        },
    },
    "bleach_bypass": {
        "id": 15,
        "energy": ("low", "high"),
        "feel": "The silver was left in the print. Contrast hardens, colour drains to "
                "almost nothing, blacks go to tar and highlights to chalk. Cold, harsh, "
                "documentary — a war photograph rather than a snapshot.",
        "use_when": "Weather, grief, stone, rain, winter light, hospital corridors, a "
                    "hard face. Also the honest answer for a badly-lit picture: it "
                    "turns muddy colour into deliberate grit.",
        "avoid_when": "Anything tender, warm or celebratory — it drains exactly the "
                      "colour those memories are made of. Also under the muted or "
                      "pastel film grade, which already pulled saturation down; the two "
                      "together leave a grey rectangle.",
        "params": {
            "strength":   (0.0, 1.0, 0.5, ("How much of the bleached print replaces the "
                                           "photograph. 0.3 is a hardened version of "
                                           "the same shot, 1.0 is a different stock.")),
            "desaturate": (0.0, 1.0, 0.45, ("How much colour is drained. 0.4 leaves "
                                            "skin recognisable, 0.9 is nearly mono with "
                                            "a memory of hue.")),
            "contrast":   (0.0, 1.0, 0.55, ("How hard the curve bites. Low is a gentle "
                                            "firming, high blows the highlights out and "
                                            "closes the shadows.")),
            "crush":      (0.0, 1.0, 0.3, ("How far the blacks are pulled under. A "
                                           "little is silver; past 0.6 the shadow "
                                           "detail is gone for good.")),
        },
    },
    "cross_process": {
        "id": 16,
        "energy": ("high",),
        "feel": "Film developed in the wrong chemistry. The curves come apart — shadows "
                "go cyan and cold, highlights go yellow and hot, colour shouts, "
                "contrast climbs. Nineties, accidental, expensive-looking by mistake.",
        "use_when": "Night, neon, parties, swimming pools, skateboards, anything "
                    "already loud. The one effect here that makes a picture look MORE "
                    "alive rather than more remembered.",
        "avoid_when": "Skin lit by a window and any quiet passage of the score — the "
                      "yellow highlight lands straight on a face and turns it "
                      "jaundiced. Never alongside duotone or infrared.",
        "params": {
            "strength":   (0.0, 1.0, 0.5, ("How much of the mis-developed print comes "
                                           "through. 0.3 is a hint of the wrong bath, "
                                           "1.0 is the whole mistake.")),
            "curve":      (0.0, 1.0, 0.55, ("How hard the S-curve bends. 0.2 is a "
                                            "slight firming, 1.0 crushes the shadows "
                                            "and blows the lights.")),
            "split":      (0.0, 1.0, 0.5, ("How far the two ends separate — cyan in the "
                                           "shadows, yellow in the lights. This is the "
                                           "look. Past 0.7 the whites turn visibly "
                                           "yellow and it reads as a fault.")),
            "saturation": (0.0, 1.0, 0.35, ("Extra colour on top. 0 keeps the "
                                            "photograph's own, 1.0 doubles it — good on "
                                            "neon, ruinous on skin.")),
        },
    },
    "infrared": {
        "id": 17,
        "energy": ("high",),
        "feel": "False colour, the way Aerochrome saw the world: every leaf burns to "
                "white or hot magenta, the sky goes to near black, everything else "
                "falls to hard monochrome. Alien, and the landscape underneath is still "
                "recognisably yours.",
        "use_when": "Green and sky in the same frame — a park, a forest, a garden, a "
                    "field, a summer walk. It needs foliage and it needs daylight; that "
                    "is the entire condition.",
        "avoid_when": "Interiors, night, winter, cities, and any close portrait — with "
                      "no foliage it is just a harsh black-and-white with a black sky. "
                      "Once per film at most; twice and it is a filter.",
        "params": {
            "strength": (0.0, 1.0, 0.4, ("How much of the false-colour film replaces "
                                         "the photograph. 0.3 is a summer haze over the "
                                         "real picture, 0.8 is another planet.")),
            "foliage":  (0.0, 1.0, 0.6, ("How far the greens blow out. 0.3 makes leaves "
                                         "pale, 1.0 puts them past white — the point of "
                                         "the effect, so keep it above 0.4.")),
            "sky":      (0.0, 1.0, 0.5, ("How dark the blues go. 0.2 is a deep summer "
                                         "sky, 1.0 is a black one behind white trees.")),
            "tint":     (0.0, 1.0, 0.88, ("Hue of the false colour on the foliage: 0 "
                                          "red, 0.1 amber, 0.55 cyan, 0.88 magenta. "
                                          "Aerochrome is magenta; anything else is an "
                                          "invention.")),
        },
    },
    "colour_isolate": {
        "id": 18,
        "energy": ("calm", "high"),
        "feel": "The whole frame falls to grey except one colour, which stays — and "
                "gets louder. Deliberate and a little theatrical: it does not change "
                "the picture, it points at part of it.",
        "use_when": "There is ONE coloured thing worth pointing at and you can name its "
                    "hue: a red coat in a crowd, a blue door, a yellow taxi, one lit "
                    "window. Strongest on the shot the film is actually about.",
        "avoid_when": "Frames where the chosen colour is everywhere (a green field, a "
                      "blue sea) — isolating it drains nothing and looks broken — and "
                      "frames where it is nowhere, which is just a mono shot. Never "
                      "after duotone, which has destroyed the hue this needs to find.",
        "params": {
            "hue":   (0.0, 1.0, 0.02, ("Which colour survives, round the wheel: 0.02 "
                                       "red, 0.1 orange, 0.15 yellow, 0.33 green, 0.6 "
                                       "blue, 0.85 magenta. Name the thing in the "
                                       "photograph and pick its hue.")),
            "width": (0.04, 0.6, 0.16, ("How wide the surviving band is. 0.05 keeps one "
                                        "exact colour and nothing beside it, 0.4 keeps "
                                        "a quarter of the wheel.")),
            "drain": (0.0, 1.0, 0.6, ("How grey everything else goes. 0.5 is a "
                                      "washed-out world with one colour in it, 1.0 is "
                                      "true black-and-white around the subject.")),
            "boost": (0.0, 1.0, 0.3, ("How much louder the surviving colour gets. 0 "
                                      "leaves it as photographed, 0.8 makes it glow. "
                                      "Above 0.5 it looks painted on.")),
        },
    },
    "flash": {
        "id": 19,
        "energy": ("high",),
        "feel": "One frame of white. The picture is blown out for an instant and then "
                "comes back over half a second, the way a room does after a camera "
                "fires. A punctuation mark you feel in your eyes.",
        "use_when": "A single hard accent: the exact frame a downbeat lands on, the "
                    "first shot of a chapter, a photograph that was taken WITH a flash, "
                    "a firework, a lightning storm. Once or twice in a whole film.",
        "avoid_when": "Anything tender or slow, and never twice in the same shot — a "
                      "picture that keeps blowing out is a fault, not an accent. Never "
                      "under bloom, which is already spilling the highlights.",
        "params": {
            "at":       (0.05, 0.95, 0.12, ("WHERE IN THE SHOT it fires, as a fraction "
                                            "of the shot's own length — not of this "
                                            "beat. Put it inside the beat's from/to "
                                            "window or it never fires. 0.05 is right on "
                                            "the cut, 0.5 is mid-shot.")),
            "strength": (0.0, 1.0, 0.7, ("How white the peak goes. 0.4 is a lamp "
                                         "passing, 1.0 is total blowout for a frame.")),
            "decay":    (0.05, 0.6, 0.25, ("Seconds of recovery. 0.1 is a camera flash, "
                                           "0.5 is a firework fading. Keep it under a "
                                           "third of the beat's length so the picture "
                                           "is fully back before the beat hands over.")),
            "warmth":   (-1.0, 1.0, 0.0, ("Colour of the blowout. 0 is a white flash, "
                                          "positive is a firework or a match, negative "
                                          "is lightning and a phone screen.")),
        },
    },
    # --- geometry: what the CAMERA does with the frame (ids 20-27) -----------------
    "pixelate": {
        "id": 20,
        "energy": ("high", "calm"),
        "feel": "The photograph is a grid of flat squares that shrink until it is a "
                "photograph again — or the other way round, breaking apart into blocks. "
                "Digital, deliberate, and it hides the picture before it gives it back.",
        "use_when": "The first shot of a chapter, a reveal you want to earn, or a face "
                    "you want the viewer to resolve rather than be handed. Also good on "
                    "a screenshot or a phone photo, anything already digital.",
        "avoid_when": "A shot that is already busy or already treated — blocks over a "
                      "crowd read as a compression fault, not a choice. Never twice in "
                      "a film.",
        "params": {
            "coarse":    (0.0, 1.0, 0.4, ("Biggest block size. 0.1 is a fine mosaic you "
                                          "can still read through, 1.0 is seven huge "
                                          "squares across the frame.")),
            "direction": (0.0, 1.0, 0.0, ("0 starts blocky and RESOLVES into the sharp "
                                          "photo; 1 starts sharp and CRUMBLES into "
                                          "blocks.")),
            "span":      (0.05, 1.0, 0.35, ("Fraction of the shot the resolve takes, "
                                            "from its START. 0.15 snaps clear at once, "
                                            "0.9 is a long slow arrival.")),
            "stretch":   (0.0, 1.0, 0.0, ("Shape of a block. 0 is square, 1 is a wide "
                                          "horizontal bar — flat strips, not a mosaic.")),
        },
    },
    "zoom_punch": {
        "id": 21,
        "energy": ("high",),
        "feel": "The camera jumps IN on the downbeat and settles back before the next "
                "one. Not a push — a hit. You feel the music in your chest.",
        "use_when": "A driving score with an obvious beat and a shot that can take it: "
                    "a crowd, a dance, a run, a strong single subject dead centre. Best "
                    "on short shots where two or three punches land and it is over.",
        "avoid_when": "Slow passages, wide landscapes and anything with a horizon you "
                      "want to stay put. Alongside pulse it doubles the beat and reads "
                      "as a fault.",
        "params": {
            "bpm":   (30.0, 180.0, 100.0, ("Beats per minute to punch on. USE THE TEMPO "
                                           "GIVEN IN THE BRIEF, or the picture hits "
                                           "against the music instead of with it.")),
            "depth": (0.0, 1.0, 0.45, ("How far in it jumps. 0.2 is a nudge, 1.0 is a "
                                       "16% snap that crops hard for a moment.")),
            "phase": (0.0, 1.0, 0.0, ("Offset onto the downbeat, in fractions of a "
                                      "beat. Leave at 0 unless the shot starts "
                                      "off-beat.")),
            "snap":  (0.0, 1.0, 0.25, ("How sharp the hit is. 0 is a hard stab gone "
                                       "instantly, 1 is a soft swell filling the beat.")),
        },
    },
    "shake": {
        "id": 22,
        "energy": ("high",),
        "feel": "The frame is held in a hand, not on a tripod — it jitters and rolls a "
                "degree or so, and it kicks harder every beat. Urgent, present, "
                "slightly out of control.",
        "use_when": "Movement and noise: a crowd, a concert, running children, a car, a "
                    "storm. Also good on a photo that is already slightly wrong — "
                    "tilted, grabbed, half-blurred — because it makes the mistake look "
                    "intended.",
        "avoid_when": "Faces you want the viewer to hold, architecture, text, and any "
                      "quiet or sad passage. Never with weave, which is the same "
                      "instability at a smaller size.",
        "params": {
            "amount": (0.0, 1.0, 0.4, ("How far it moves. 0.2 is a steady hand, 1.0 is "
                                       "about 4% of the frame and reads as a scramble.")),
            "rate":   (0.0, 1.0, 0.45, ("How fast the jitter is. 0 is a slow float, 1 "
                                        "is a nervous buzz.")),
            "bpm":    (30.0, 180.0, 100.0, ("Beats per minute for the kick. USE THE "
                                            "TEMPO GIVEN IN THE BRIEF.")),
            "accent": (0.0, 1.0, 0.5, ("How much of the shake is only ON the beat. 0 is "
                                       "a constant handheld bed, 1 goes still between "
                                       "hits and stabs on each one.")),
        },
    },
    "dolly_swing": {
        "id": 23,
        "energy": ("calm", "low"),
        "feel": "The camera walks INTO the picture and leans sideways as it goes, then "
                "eases back out. A move, not an effect — you read it as the operator "
                "stepping forward rather than as something done to the photo.",
        "use_when": "A room, a table, a street, a group — anywhere there is depth to "
                    "walk into and a long enough shot to do it slowly. The calm answer "
                    "when a shot needs motion but not energy.",
        "avoid_when": "Very short shots, where the swing has no room and reads as a "
                      "wobble. Also on a tight crop, already as far in as the frame "
                      "goes.",
        "params": {
            "push":   (0.0, 1.0, 0.5, ("How far it walks in. 0.3 is a step, 1.0 is a "
                                       "12% push on top of the camera's own move.")),
            "drift":  (0.0, 1.0, 0.45, ("How far it leans sideways. 0.2 is a shift of "
                                        "weight, 1.0 is 3.5% of the frame.")),
            "angle":  (0.0, 1.0, 0.0, ("Which way it leans, as a fraction of a full "
                                       "turn. 0 goes right, 0.25 up, 0.5 left.")),
            "swings": (0.0, 1.0, 0.0, ("How many times it leans and returns. 0 is one "
                                       "single lean out and back; 1 is four, a slow "
                                       "pendulum.")),
        },
    },
    "mirror": {
        "id": 24,
        "energy": ("high", "calm"),
        "feel": "A fold sweeps in from the edge and the far side of the frame becomes a "
                "reflection of the near side. Two ways is a butterfly; four ways is one "
                "quarter of the photo repeated into a tile. Graphic and unreal.",
        "use_when": "Symmetry that is already there — a face, a doorway, a reflection "
                    "in water, a road running away from you. Also on an abstract or a "
                    "texture, where there is no subject to break.",
        "avoid_when": "Groups and candid shots: mirroring puts half a person against "
                      "half a person and reads as a rendering bug. Never on the shot "
                      "the film is actually about.",
        "params": {
            "fold_x": (0.0, 1.0, 1.0, ("Left-right fold. 0 is none, 1 brings the axis "
                                       "to the centre for a true half-mirror.")),
            "fold_y": (0.0, 1.0, 0.0, ("Top-bottom fold. Leave at 0 for a two-way "
                                       "mirror; raise it as well for a four-way tile.")),
            "side":   (0.0, 1.0, 0.0, ("Which half survives. Below 0.5 keeps the LEFT "
                                       "and top; above 0.5 keeps the right and bottom. "
                                       "Keep the half the subject is in.")),
            "hold":   (0.0, 1.0, 0.6, ("0 sweeps the fold in and straight back out like "
                                       "a breath; 1 folds fast, holds it, and opens "
                                       "again at the end.")),
        },
    },
    "kaleidoscope": {
        "id": 25,
        "energy": ("high",),
        "feel": "The frame folds around its own centre into repeating wedges that turn "
                "slowly, like looking down a toy kaleidoscope. The middle can be left "
                "alone, so the subject stays real while the edges shatter into copies.",
        "use_when": "Flowers, water, leaves, lights, fabric, fireworks — colour and "
                    "pattern with no single subject. The strongest thing in this "
                    "library; it belongs on one shot in a film, usually the most "
                    "abstract one.",
        "avoid_when": "Faces, groups, buildings, anything with text. A person cut into "
                      "six wedges is not a memory of that person.",
        "params": {
            "segments": (0.0, 1.0, 0.4, ("How many repeats. 0 is two big halves, 0.4 is "
                                         "six wedges, 1.0 is twelve and the picture "
                                         "stops being recognisable.")),
            "strength": (0.0, 1.0, 0.7, ("How completely the fold takes over. 0.3 "
                                         "leaves the photo readable through it, 1.0 is "
                                         "a full kaleidoscope.")),
            "core":     (0.0, 1.0, 0.5, ("How much of the CENTRE stays the real "
                                         "photograph. 0 folds everything, 1 leaves a "
                                         "wide untouched disc — use a high value if "
                                         "there is a subject to protect.")),
            "hold":     (0.0, 1.0, 0.5, ("0 opens and closes the fold across the whole "
                                         "shot like a breath; 1 arrives quickly, holds, "
                                         "and unfolds at the end.")),
        },
    },
    "tilt": {
        "id": 26,
        "energy": ("low", "calm"),
        "feel": "The horizon rolls a few degrees off level and slowly comes back. "
                "Nothing else moves. It makes a calm picture feel subtly wrong, which "
                "is usually what you want from a memory you are not sure about.",
        "use_when": "Unease, drink, tiredness, late night, a long day ending. Also any "
                    "shot with a strong horizontal — a horizon, a table, a skyline — "
                    "because that is what makes the roll legible.",
        "avoid_when": "Portraits shot straight on, and anywhere shake is already "
                      "running: they are both the camera not being level, and together "
                      "they cancel into mush.",
        "params": {
            "angle":  (-1.0, 1.0, 0.4, ("How far it rolls, and which way. Positive "
                                        "rolls clockwise, negative anticlockwise; 1.0 "
                                        "is about 11 degrees, which is a lot.")),
            "zoom":   (0.0, 1.0, 0.5, ("How much it pushes in to cover the corners the "
                                       "roll opens up. Raise it if the blurred backdrop "
                                       "starts showing at the edges.")),
            "hold":   (0.0, 1.0, 0.5, ("0 rolls over and back across the whole shot; 1 "
                                       "arrives at the angle quickly, sits there, and "
                                       "levels off before the cut.")),
            "wobble": (0.0, 1.0, 0.25, ("A slow breath on top of the roll, so it feels "
                                        "held rather than set. 0 is a locked angle, 1 "
                                        "drifts a third of the way back and forth.")),
        },
    },
    "parallax": {
        "id": 27,
        "energy": ("calm", "low"),
        "feel": "The middle of the frame slides while the edges stay almost still, so "
                "the photograph separates into a near thing and a far thing. A flat "
                "picture briefly has depth, and then it settles back.",
        "use_when": "A clear subject in front of a background that is further away: a "
                    "person against a street, a tree against a sky, a table in a room. "
                    "The quietest way to make a still photo look three-dimensional.",
        "avoid_when": "Straight lines through the middle of the frame — window frames, "
                      "doorways, horizons, text — which bend as the near plane passes "
                      "over them. Also flat textures with no depth to fake.",
        "params": {
            "depth":  (0.0, 1.0, 0.45, ("How far the near plane leads the far one. 0.2 "
                                        "is a hint of depth, 1.0 is 4.5% of the frame "
                                        "and the picture starts to look rubbery.")),
            "angle":  (0.0, 1.0, 0.0, ("Which way it slides, as a fraction of a full "
                                       "turn. 0 goes right, 0.25 up, 0.5 left.")),
            "size":   (0.0, 1.0, 0.45, ("How big the near plane is. 0 is a small centre "
                                        "crop, 1 is most of the frame with only the "
                                        "outer border held back. Put it around the "
                                        "subject.")),
            "passes": (0.0, 1.0, 0.0, ("How many times it slides out and returns. 0 is "
                                       "one slow move, 1 is three and reads as a "
                                       "float.")),
        },
    },
    # --- the surface it is shown on: film, tape, tube, press, paper (ids 28-32) ----
    "film_burn": {
        "id": 28,
        "energy": ("low", "calm"),
        "feel": "A print running through a tired projector. The gate wanders and now "
                "and then loses its grip and hops; the lamp will not hold an exposure; "
                "dust, specks and a hair come and go; the edge of the frame is scorched "
                "brown with a hot rim. Weave's older, louder brother.",
        "use_when": "The oldest memories in the film — childhood, a wedding, someone "
                    "who is gone. Anything you want to feel FOUND in a box rather than "
                    "opened on a phone. Also the first or last shot, where it reads as "
                    "the reel starting or ending.",
        "avoid_when": "Alongside weave, vhs or glitch — they are the same sentence said "
                      "four ways. Never on a recent, sharp, happy photo: burning the "
                      "edges off last summer is a lie about its age.",
        "params": {
            "jitter":  (0.0, 1.0, 0.4, ("How loose the gate is. 0.2 is a well-kept "
                                        "projector, 1.0 hops hard enough to read as a "
                                        "fault. Stay under 0.6 for a memory.")),
            "flicker": (0.0, 1.0, 0.35, ("Exposure instability. 0.2 is a warm breathing "
                                         "lamp, 0.8 is a strobe you cannot look at.")),
            "dust":    (0.0, 1.0, 0.4, ("Specks, scratches and hairs. 0.2 is a clean "
                                        "print, 0.8 is one shown a hundred times.")),
            "burn":    (0.0, 1.0, 0.3, ("How far the scorched edge eats in. 0 is an "
                                        "intact frame, 0.8 is a print about to go.")),
        },
    },
    "vhs": {
        "id": 29,
        "energy": ("high", "calm"),
        "feel": "A tape, played on a machine that has played it too often. Scanlines "
                "sit over the picture, the tracking will not lock so the lines shear "
                "sideways, the foot of the frame tears, and the colour lags and smears "
                "to the right of every edge. Loud, cheap, instantly the nineties.",
        "use_when": "Home video: birthdays, camcorder holidays, kids, anything shot "
                    "between 1985 and 2003 or that you want to FEEL like it was. Very "
                    "strong on the first shot of a chapter that jumps back in time.",
        "avoid_when": "Faces you want the viewer to fall in love with — the smear lands "
                      "on eyes and mouths first. Never with film_burn, weave or crt: "
                      "tape and film are two different lies about one picture.",
        "params": {
            "tracking": (0.0, 0.7, 0.4, ("How badly the tape holds its line. 0.2 is a "
                                         "gentle waver, 0.7 tears the bottom of the "
                                         "frame open on every pass.")),
            "bleed":    (0.0, 1.0, 0.45, ("Chroma smear off each edge. 0.2 is a soft "
                                          "colour, 0.9 drags red and blue visibly "
                                          "sideways. It also buys texture samples — "
                                          "high is the only expensive setting here.")),
            "lines":    (0.0, 1.0, 0.5, ("Scanline depth. 0.2 is a suggestion, 0.9 is "
                                         "watching through a comb.")),
            "noise":    (0.0, 1.0, 0.35, ("Hiss, snow and dropout streaks. 0.2 is a "
                                          "good tape, 0.8 is one left in a hot car.")),
        },
    },
    "crt": {
        "id": 30,
        "energy": ("high",),
        "feel": "The photograph is on a television, and the television is being filmed. "
                "The glass bulges, red green and blue phosphor stripes sit under "
                "everything, the bright parts glow into their neighbours, and a wide "
                "faint bar creeps slowly up the screen.",
        "use_when": "A memory that was watched rather than lived: a match, a broadcast, "
                    "a game, a video call, someone on a screen. Also the right joke for "
                    "a shot the film is quoting rather than showing.",
        "avoid_when": "Any photograph with small text or fine detail — the grille eats "
                      "it. Never with vhs: one is the tape and the other is the set, "
                      "and running both just makes the frame dirty.",
        "params": {
            "curve": (0.0, 1.0, 0.45, ("How far the glass bulges, and how dark the "
                                       "corners of the bezel go with it. 0.2 is a flat "
                                       "late tube, 1.0 is a fishbowl.")),
            "grid":  (0.0, 1.0, 0.5, ("Phosphor stripes and scan gaps. 0.2 is a good "
                                      "monitor across a room, 0.9 is your nose against "
                                      "the glass.")),
            "roll":  (0.0, 1.0, 0.35, ("The roll bar and the vertical hold. 0 is a "
                                       "locked picture, 0.8 is a set that needs "
                                       "hitting.")),
            "glow":  (0.0, 1.0, 0.4, ("How much the phosphors bleed light back. It is "
                                      "what stops the grille from just darkening the "
                                      "picture, so keep it near grid.")),
        },
    },
    "halftone": {
        "id": 31,
        "energy": ("calm", "high"),
        "feel": "The photograph as it would come off a press: broken into a grid of "
                "round dots on off-white paper, big and open in the shadows, shrinking "
                "to nothing in the highlights. Graphic, flattened, printed.",
        "use_when": "A shot you want to read as a POSTER or a page rather than a "
                    "photograph — a title shot, a portrait held long, a picture the "
                    "film is quoting. Big dots suit bold, funny, pop moments; fine dots "
                    "suit a newspaper, an archive, something official.",
        "avoid_when": "Faces at conversational size and anything with fine detail — the "
                      "dots eat expression first. Never twice in one film, and never "
                      "under bloom, which fills the highlights the dots need empty.",
        "params": {
            "size":   (0.0, 1.0, 0.35, ("How big the dots are. 0.1 is fine newsprint "
                                        "you read through, 0.9 is comic-book dots you "
                                        "read INSTEAD of the picture.")),
            "angle":  (0.0, 1.0, 0.5, ("Rotation of the screen. 0 is straight up and "
                                       "down, 1 is a quarter turn. 0.5 is the 45 "
                                       "degrees every real press uses and the safest "
                                       "answer.")),
            "ink":    (0.0, 1.0, 0.7, ("How far the photograph is replaced by the "
                                       "print. 0.3 is a texture over the picture, 1.0 "
                                       "IS the print and nothing else.")),
            "colour": (0.0, 1.0, 0.4, ("What the ink is. 0 is black on paper — "
                                       "newspaper, documentary, old. 1 keeps the "
                                       "photograph's own colours in the dots, which is "
                                       "bright and comic-strip.")),
        },
    },
    "paper": {
        "id": 32,
        "energy": ("low", "calm"),
        "feel": "Not a photograph — a PRINT of one, lying on a table. The sheet lifts "
                "and bows at its right edge and shades where it turns from the light, "
                "the border is torn ragged with a browned lip, and the fibre of the "
                "stock shows through the picture and does not move.",
        "use_when": "The most personal shots in the film, and the oldest: a single "
                    "face, a letter, a keepsake, the last shot before the credits. The "
                    "quietest way in this library to say a memory is an OBJECT somebody "
                    "kept.",
        "avoid_when": "Bright, busy, recent, moving moments — a torn edge on a photo "
                      "taken last week is a costume. Never with film_burn: they are the "
                      "same idea, and stacked they read as damage rather than age.",
        "params": {
            "curl":  (0.0, 1.0, 0.4, ("How far the right edge lifts and shades. 0.2 is "
                                      "a sheet that has been in an album, 1.0 is one "
                                      "that was rolled up in a drawer.")),
            "tear":  (0.0, 1.0, 0.45, ("How ragged the border is. 0 is a clean machine "
                                       "cut, 0.4 is torn by hand, 1.0 is torn and "
                                       "singed.")),
            "grain": (0.0, 1.0, 0.4, ("Paper fibre showing through. 0.2 is a gloss "
                                      "print, 0.9 is thick matte stock you can feel.")),
            "age":   (0.0, 1.0, 0.45, ("How old the stock is. 0 is white and recent, "
                                       "1.0 is yellowed, and the torn lip goes dark "
                                       "brown with it.")),
        },
    },
    # --- rhythm: effects that happen ON the beat (ids 33-35) -----------------------
    "strobe_cut": {
        "id": 33,
        "energy": ("high",),
        "feel": "The shot holds perfectly still, and then the frame is struck: two or "
                "three bright frames in a row on the downbeat, each weaker than the one "
                "before, and it is clean again before you have finished seeing it.",
        "use_when": "Loud, fast, celebratory moments where the score has a hard "
                    "downbeat — dancing, a crowd, fireworks, the middle of a party. The "
                    "only effect here that hits ON the beat rather than breathing with "
                    "it, so put it exactly where the music lands.",
        "avoid_when": "Anything quiet or sad; any shot long enough for the burst to come "
                      "round twice, where it stops being an accent and becomes a "
                      "nightclub; and any face you want read, because the flash blows "
                      "out skin. Never with pulse — that is the same idea twice.",
        "params": {
            "bpm":    (30.0, 180.0, 96.0, ("Beats per minute the flashes land on. USE "
                                           "THE TEMPO GIVEN IN THE BRIEF, or the "
                                           "picture hits between the beats instead of "
                                           "on them.")),
            "frames": (1.0, 4.0, 3.0, ("How many bright frames each hit fires. 1 is a "
                                       "single stab, 3 is the classic stutter, 4 starts "
                                       "to look like a fault.")),
            "depth":  (0.0, 1.0, 0.55, ("How hard the frame is struck. 0.3 is a camera "
                                        "flash across the room, 0.9 blows the picture "
                                        "to white.")),
            "hold":   (0.05, 0.9, 0.35, ("Fraction of the shot held perfectly clean "
                                         "before the FIRST hit — and the hit lands "
                                         "exactly there, so put it on the downbeat you "
                                         "want.")),
        },
    },
    "freeze_ripple": {
        "id": 34,
        "energy": ("calm", "high"),
        "feel": "One wave crosses the photograph and is gone. It swells out of a point, "
                "bends everything it passes through for a moment, dies at the corners, "
                "and does not come back.",
        "use_when": "A single moment inside an otherwise still shot: the beat a name is "
                    "spoken on, the instant a memory lands, water, glass, heat, a "
                    "shock. Because it happens exactly ONCE it is punctuation — centre "
                    "it on the subject and let the rest of the shot be quiet.",
        "avoid_when": "Straight architecture and text, which visibly bend as it passes; "
                      "and shots under about two seconds, where the wave has no room to "
                      "travel and just reads as a wobble.",
        "params": {
            "x":      (0.0, 1.0, 0.5, ("Where the wave is born, across the frame. 0 is "
                                       "the left edge, 1 the right. Put it ON the thing "
                                       "the shot is about.")),
            "y":      (0.0, 1.0, 0.5, ("Where it is born down the frame. 0 is the TOP, "
                                       "1 the bottom. Faces usually sit near 0.35.")),
            "amount": (0.0, 1.0, 0.45, ("How hard the wave bends what it passes "
                                        "through. 0.2 is a breath on water, 0.9 is a "
                                        "shockwave.")),
            "at":     (0.0, 1.0, 0.25, ("When it is released, as a fraction of the "
                                        "shot. Put it on a downbeat — this is the only "
                                        "moment the effect exists.")),
        },
    },
    "echo": {
        "id": 35,
        "energy": ("high", "calm"),
        "feel": "The picture leaves copies of itself behind — a short decaying trail of "
                "ghosts in the direction it is travelling, brightest nearest the frame "
                "and gone a few steps back. Video feedback, not motion blur: only the "
                "lights smear, so the photograph underneath stays readable.",
        "use_when": "Movement and momentum: a dance floor, a fairground, headlights, "
                    "rain, anything at night with points of light in it, and any shot "
                    "whose camera push you want the viewer to FEEL rather than notice. "
                    "It is a bed, so it can hold under a whole shot.",
        "avoid_when": "Faces and anything you want read in detail — the ghosts sit on "
                      "top of it. Also flat daylight, where nothing is bright enough to "
                      "trail and it only greys the picture down.",
        "params": {
            "spread": (0.0, 1.0, 0.4, ("How far back the trail reaches. 0 is off "
                                       "entirely, 0.3 is a shimmer behind the lights, "
                                       "1.0 is a long smear.")),
            "decay":  (0.05, 0.95, 0.55, ("How fast each ghost fades behind the one in "
                                          "front. 0.1 is a long tail of many, 0.9 is "
                                          "one close ghost.")),
            "angle":  (0.0, 1.0, 0.0, ("Direction the ghosts slide, as a fraction of a "
                                       "full turn. 0 trails to the right, 0.25 "
                                       "upward.")),
            "zoom":   (-1.0, 1.0, 0.5, ("How much of the trail is a zoom instead of a "
                                        "slide. Positive trails a push IN, negative a "
                                        "pull back, 0 is a pure sideways smear.")),
        },
    },
}
DEFAULT_SHOT = "plain"

TRANSITIONS: dict[str, dict] = {
    "none": {
        "id": 0, "seconds": 0.0,
        "feel": "The first shot. There is nothing to come from.",
        "use_when": "Only shot 1. Never choose this.",
    },
    "dissolve": {
        "id": 0, "seconds": 0.8,
        "feel": "One picture fades through into the next; for a moment both are there.",
        "use_when": "The default. Two moments that belong to each other — same place, "
                    "same few minutes, same feeling.",
    },
    "cut": {
        "id": 1, "seconds": 0.0,
        "feel": "Instant. No blend at all.",
        "use_when": "Retakes of the same frame, and any moment you want to land exactly "
                    "ON a beat — a dissolve smears the beat, a cut hits it.",
    },
    "dip": {
        "id": 2, "seconds": 1.1,
        "feel": "Down to black and back up. A breath between two things.",
        "use_when": "A real break: a new place, hours later, a change of subject. It is "
                    "the strongest punctuation here — use it where a chapter ends.",
    },
    "whip": {
        "id": 3, "seconds": 0.45,
        "feel": "Both frames smear radially, peaking mid-cut, then snap clear. Fast and "
                "physical, like the camera was thrown.",
        "use_when": "Energy: a run of quick shots, a strong downbeat, a montage picking "
                    "up speed. Never twice in a row — it becomes a tic.",
    },
    "slide": {
        "id": 5, "seconds": 0.7,
        "feel": "The outgoing photograph is PUSHED off to the left as the next arrives "
                "from the right, both travelling together like frames on a strip.",
        "use_when": "Pictures that belong side by side — a walk down a street, a row of "
                    "faces, things seen one after another. It reads as turning a page, "
                    "so it suits moving ALONG rather than moving on.",
    },
    "slide_up": {
        "id": 6, "seconds": 0.7,
        "feel": "The outgoing photograph is pushed UP out of frame while the next "
                "rises from below.",
        "use_when": "Going up or down — stairs, a mountain, a view from a window — and "
                    "for a change of subject that should feel like a step rather than "
                    "a page turn. Suits portraits, which fill the frame vertically.",
    },
    "burn": {
        "id": 4, "seconds": 1.0,
        "feel": "The outgoing picture dissolves in patches on a noise field, like film "
                "burning through from the middle.",
        "use_when": "Once or twice in a whole film, on the most emotional joins. It is "
                    "the most distinctive transition here and it spends itself fast.",
    },
    "zoom_blur": {
        "id": 7, "seconds": 0.5,
        "feel": "The camera lunges. Both frames stretch into radial streaks out of the "
                "centre, the photograph underneath changes while nothing is readable, "
                "and it snaps clear on the far side.",
        "use_when": "Landing hard on a downbeat in a fast passage, or joining two shots "
                    "of the SAME subject at different distances — the streak hides the "
                    "join, so the second photograph feels arrived at rather than cut "
                    "to. The fastest thing in this list; spend it where the film "
                    "accelerates.",
    },
    "glitch_cut": {
        "id": 8, "seconds": 0.25,
        "feel": "A hard cut with two frames of damage over the join: rows slip "
                "sideways, the picture coarsens into blocks, and for an instant the two "
                "photographs are interleaved in bands.",
        "use_when": "A jolt — the exact frame a beat lands on, the first shot of a "
                    "harder chapter, a burst of near-identical retakes you want to feel "
                    "frantic. As rare as the glitch shot effect: once or twice in a "
                    "whole film, and never on the same join as one.",
    },
    "iris": {
        "id": 9, "seconds": 0.9,
        "feel": "A circle. The next picture opens out of a point at the centre — or the "
                "outgoing one closes into one — with a thin bright rim riding the edge "
                "as it travels. Old-fashioned, deliberate, and very quiet.",
        "use_when": "Singling something out, or letting it go: the shot that introduces "
                    "a person, the last shot before a chapter ends, a memory that "
                    "narrows to one detail. Which way the circle travels is not yours "
                    "to choose — it follows the incoming shot's own camera move.",
    },
    "page_turn": {
        "id": 10, "seconds": 1.0,
        "feel": "The outgoing photograph lifts at one edge and peels across the frame, "
                "showing its own dim back as it folds, its shadow running ahead of it "
                "over the picture underneath.",
        "use_when": "An album. Photographs that are separate PAGES of one story — a new "
                    "day, a new place, the next of a set — rather than two moments "
                    "inside one. The most literal transition here, which is exactly why "
                    "it should appear two or three times at most.",
    },
    "shatter": {
        "id": 11, "seconds": 1.0,
        "feel": "The outgoing frame breaks into tiles that let go one after another and "
                "fall out of the picture, darkening as they drop, until only the next "
                "photograph is left standing.",
        "use_when": "A loss or a break — the end of something, a memory you want to "
                    "feel given up rather than moved on from. Strong, and it reads as "
                    "damage, so it belongs at most once in a film and never on a tender "
                    "join.",
    },
    "match_zoom": {
        "id": 12, "seconds": 0.9,
        "feel": "One continuous push that begins inside the first photograph and "
                "finishes inside the second: the outgoing frame keeps growing, the "
                "incoming one arrives out of depth at exactly the same rate, and the "
                "eye reads a single camera move rather than a change of picture.",
        "use_when": "Two photographs that RHYME — the same subject nearer and further, "
                    "the same shape in two different places, a face and then the thing "
                    "it is looking at. The only transition that makes two pictures feel "
                    "like one shot, so spend it on the join the memory is about.",
    },
}
DEFAULT_TRANSITION = "dissolve"

PARAM_ORDER: dict[str, tuple[str, ...]] = {
    name: tuple(entry["params"]) for name, entry in SHOT_EFFECTS.items()
}
PARAM_SLOTS = 4

# --- effects over TIME -------------------------------------------------------------
#
# One treatment held for eight seconds is a screensaver. A shot is a little arc, so it
# carries a SEQUENCE of beats — `{effect, params, from, to}` with `from`/`to` as
# fractions of the shot — and they are allowed to OVERLAP, which is how an effect hands
# over to the next instead of snapping.
#
# The renderer composes at most `MAX_ACTIVE` of them at once. Two is not an arbitrary
# cap: it is one thing you notice plus one thing you feel, and past that the frame
# stops being a photograph and starts being a demo.
MAX_ACTIVE = 2
# The most beats a single shot may carry. Enough for "arrive, settle, leave".
MAX_BEATS = 4
# Every beat is at least this long (as a fraction of the shot) — anything briefer than
# a blink registers as a glitch rather than as an effect.
MIN_BEAT_SPAN = 0.12


def pack_beats(raw, fallback: str, fallback_params: dict | None = None) -> list[dict]:
    """Validate a shot's effect SEQUENCE into `{effect, params, from, to}` entries.

    Everything is repaired rather than rejected: unknown names fall back, numbers are
    clamped, `from`/`to` are ordered and pulled inside `[0, 1]`, too-short beats are
    widened, and a missing or empty list becomes a single full-length beat of
    `fallback`. A film never loses a shot to a malformed sequence.
    """
    entries = raw if isinstance(raw, list) else []
    beats: list[dict] = []
    for entry in entries[:MAX_BEATS]:
        if not isinstance(entry, dict):
            continue
        effect = shot_effect(entry.get("effect"), fallback=fallback)
        start = _fraction(entry.get("from"), 0.0)
        end = _fraction(entry.get("to"), 1.0)
        if end < start:
            start, end = end, start
        if end - start < MIN_BEAT_SPAN:
            end = min(1.0, start + MIN_BEAT_SPAN)
            start = min(start, max(0.0, end - MIN_BEAT_SPAN))
        beats.append({
            "effect": effect,
            "params": pack_params(effect, entry.get("params")),
            "from": round(start, 3),
            "to": round(end, 3),
        })
    if not beats:
        beats = [{
            "effect": shot_effect(fallback),
            "params": pack_params(fallback, fallback_params or defaults(fallback)),
            "from": 0.0,
            "to": 1.0,
        }]
    return sorted(beats, key=lambda beat: (beat["from"], beat["to"]))


def _fraction(value, default: float) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def shot_effect(name: str | None, *, fallback: str = DEFAULT_SHOT) -> str:
    """A valid shot-effect name. Anything unknown becomes `fallback`."""
    return name if name in SHOT_EFFECTS else fallback


def transition(name: str | None, *, fallback: str = DEFAULT_TRANSITION) -> str:
    return name if name in TRANSITIONS else fallback


def transition_seconds(name: str) -> float:
    return TRANSITIONS[transition(name)]["seconds"]


def pack_params(name: str, raw: dict | None) -> list[float]:
    """The effect's parameters as the `vec4` the shader reads.

    Missing numbers take the catalogue default, out-of-range numbers are CLAMPED, and
    anything unparseable falls back too.
    """
    spec = SHOT_EFFECTS[shot_effect(name)]["params"]
    values = raw if isinstance(raw, dict) else {}
    packed: list[float] = []
    for key, (low, high, default, _note) in spec.items():
        try:
            value = float(values.get(key, default))
        except (TypeError, ValueError):
            value = default
        packed.append(round(min(max(value, low), high), 4))
    return packed + [0.0] * (PARAM_SLOTS - len(packed))


def defaults(name: str) -> dict:
    """The catalogue's own answer for an effect, for the director's fallback path."""
    return {key: spec[2]
            for key, spec in SHOT_EFFECTS[shot_effect(name)]["params"].items()}


def reachable(energy: str | None = None) -> dict[str, dict]:
    """The effects a memory of this energy may use.

    Every entry declares the moods it belongs to, because a library of thirty is a
    library the conductor cannot hold in its head OR in its context — the full table is
    2 871 tokens today and the brief already carries a line per photograph. A birthday
    is never offered `soft_focus`; a hospital visit is never offered `kaleidoscope`.
    """
    if energy is None:
        return dict(SHOT_EFFECTS)
    return {name: entry for name, entry in SHOT_EFFECTS.items()
            if energy in entry.get("energy", ("high", "calm", "low"))}


def catalogue_for_prompt(energy: str | None = None) -> str:
    """The library as the model sees it: what each effect FEELS like, when to reach for
    it, when not to, and what every number does at each end of its range.

    Generated from the same dict the shader switches on, so the prompt can never
    advertise an effect or a parameter that does not exist. `energy` narrows it to the
    effects that suit this memory (see `reachable`).
    """
    lines = ["SHOT EFFECTS — pick ONE per shot and give its parameters:", ""]
    for name, entry in reachable(energy).items():
        lines.append(f"### {name}")
        lines.append(f"  Looks like: {entry['feel']}")
        lines.append(f"  Use when:   {entry['use_when']}")
        lines.append(f"  Avoid when: {entry['avoid_when']}")
        if entry["params"]:
            lines.append("  Parameters:")
            for key, (low, high, default, note) in entry["params"].items():
                lines.append(f"    {key} ({low}–{high}, default {default}): {note}")
        else:
            lines.append("  Parameters: none.")
        lines.append("")
    lines.append("TRANSITIONS — how each shot ARRIVES (shot 1 needs none):")
    lines.append("")
    for name, entry in TRANSITIONS.items():
        if name == "none":
            continue
        lines.append(f"### {name} ({entry['seconds']}s)")
        lines.append(f"  Looks like: {entry['feel']}")
        lines.append(f"  Use when:   {entry['use_when']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def payload() -> dict:
    """What the timeline JSON carries so the renderer needs no duplicated table."""
    return {
        "shot": {name: entry["id"] for name, entry in SHOT_EFFECTS.items()},
        "transition": {name: entry["id"] for name, entry in TRANSITIONS.items()},
        "max_active": MAX_ACTIVE,
    }
