# Listening rubric (draft 2)

Clips are 32-bar excerpts rendered with the same piano sound. A section is 8 bars, so a clip has 4 sections, each about a quarter of the clip. Scores are 1-5 in steps of 0.5; 2 and 4 sit between the written anchors.

Do not score: the first and last bar (clips are cut from longer pieces; use the tags), the piano sound itself, or the tempo unless it is marked with the "odd tempo" tag.

Each criterion covers one thing. If a problem fits two criteria, score it in the one named under "Not here".

## Harmony

Do the notes agree with each other and with a key?

- **Scope:** listen section by section, then give the clip the score of a typical section, lowered by 0.5 if one section is clearly worse.
- **Listen for:** melody notes that clash with the chord under them, piano notes that clash with each other, chords that do not belong to the key, a key that feels lost.
- **Not here:** a melody that is dull but correct (Melody), a sudden stop (Flow).
- **5:** no wrong-sounding notes; the key is always clear; chord changes sound intended.
- **4:** one or two brief clashes that pass quickly.
- **3:** a few clashes, or one chord that sounds wrong, but the key stays clear.
- **2:** clashes in most sections; the key drifts.
- **1:** constant clashes; no key can be heard.

## Melody

Is there a tune on top that you could sing or hum?

- **Scope:** whole clip.
- **Listen for:** a line that stands out above the piano, steps and leaps that a voice could sing, a short motif (a few notes) that comes back, gaps where the tune disappears.
- **Not here:** clashes with the piano (Harmony), how phrases are shaped (Phrasing).
- **5:** a clear, singable tune through the whole clip, with a motif you could hum afterwards.
- **4:** clear and singable, but no motif sticks.
- **3:** a tune is there but wanders, or drops out for more than 2 bars once or twice.
- **2:** the tune is hard to follow: too many leaps, too high or low, or missing for long stretches.
- **1:** no tune can be heard, or the top notes are random.

## Flow

Does each bar lead naturally into the next?

- **Scope:** bar to bar, across the whole clip.
- **Listen for:** sudden stops, silent bars that were not prepared, jumps in density (busy bar then near-empty bar), notes that feel late or early against the beat, melody and piano drifting apart in time, the pace feeling like it changes.
- **Not here:** whether phrases sound like question and answer (Phrasing), the big-picture shape (Form).
- **5:** every bar joins smoothly; changes in density or register feel prepared.
- **4:** one awkward join.
- **3:** two or three awkward joins, or one silent gap.
- **2:** awkward joins in most sections; it often feels like it stumbles.
- **1:** sounds like unrelated bars stitched together.

## Phrasing

Inside each 8-bar section, do the notes group into phrases?

- **Scope:** within each 8-bar section; score how many of the 4 sections have phrases.
- **Listen for:** a 4-bar idea followed by a 4-bar answer, a 2- or 4-bar idea repeated with a change, phrase endings that feel like a breath or a comma (a longer note, a pause, a cadence such as V to I).
- **Not here:** whether sections relate to each other (Form).
- **5:** all 4 sections have clear phrases with endings you can feel.
- **4:** 3 sections do.
- **3:** 2 sections do, or phrases are there but endings are weak.
- **2:** 1 section does.
- **1:** notes run on with no phrase shape anywhere.

## Form

Across all 32 bars, does the music go somewhere?

- **Scope:** whole clip, comparing the 4 sections with each other.
- **Listen for:** a section that contrasts (new melody, new piano pattern, higher or fuller), earlier material coming back (A ... B ... A'), a build in energy, a section that sounds like a chorus. Ignore that the clip is cut at the start and end.
- **Not here:** phrase shape inside a section (Phrasing), smoothness between bars (Flow).
- **5:** clear contrast and a return, or a clear build; you could sketch the shape (for example A A B A).
- **4:** clear contrast, but nothing returns.
- **3:** one section stands out a little; the rest sounds alike.
- **2:** the same texture all the way, or changes that feel random.
- **1:** no shape at all, or it keeps changing with no connection.

## Accompaniment

Does the piano part support the melody?

- **Scope:** whole clip.
- **Listen for:** a bass line in the left hand, chords or broken chords that match the melody's pace, fills in the melody's gaps, variation between sections, the piano covering or fighting the melody.
- **Not here:** wrong notes (Harmony).
- **5:** varied and fitting: bass, chords and fills change with the music and leave room for the melody.
- **4:** fitting, with a little variation.
- **3:** fitting but the same pattern throughout.
- **2:** too thin (single notes, long empty stretches) or too busy (covers the melody).
- **1:** fights the melody or sounds unrelated to it.

## Overall

Your own judgement as a listener, everything included.

- **Scope:** whole clip.
- **5:** I would gladly hear it again; could pass for a real pop piano piece.
- **4:** good, with small issues.
- **3:** okay; noticeable issues but still music.
- **2:** hard to enjoy; issues dominate.
- **1:** unpleasant or not music.

## Issue tags

Tick any that apply; they are counted per source in the report.

- **Silent gap:** a stretch of a bar or more with no sound inside the clip.
- **Abrupt start:** the first bars sound random or unprepared.
- **Cut-off ending:** stops mid-phrase.
- **Wrong notes:** at least one clearly wrong note.
- **Too repetitive:** the same bar or pattern many times with no change.
- **Odd tempo:** much too slow or fast for pop, or the pace seems to change.
- **Melody missing:** no melody for 4 bars or more.
- **Timing off:** melody and piano not aligned in time.

## A/B pages: element strips (from `t3_continuations`)

Under the preference, each piece has an optional "Details" panel: the six elements above, each marked weak, ok or strong. Choosing a level opens four bar ranges (1-8, 9-16, 17-24, 25-32) for where it applies; none selected means the whole piece. `listening.py reveal` writes `results/listening/<session>_elements.csv` (system, element, level, bars) and prints the mean level per system and element. In TASK B sessions (`listening.py prompt_pairs`) bars 1-8 are the prompt, and `reveal` also prints the wins per prompt and config.
