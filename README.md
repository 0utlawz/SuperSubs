# Ball possession from automatic tracking — More than a Hack 2027

For every frame of FC Barcelona – Athletic Club, say who has the ball (`H` / `A`) or that it is out
of play (`D`). Submission for [More than a Hack 2027, Phase 1](https://github.com/metrica-sports/more-than-a-hack-2027).

## Reproduce the submission

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py                             # -> outputs/SuperSubs.csv
```

The match data is not in this repository and must never be: it is licensed for the competition
only. The code expects the official repository cloned next to this one with the data unpacked in
its `data/` folder. Set `MTAH_DATA_DIR` (and `MTAH_REPO` for scoring) to point elsewhere.

```bash
.venv/bin/python score.py outputs/SuperSubs.csv     # official validator + public 5-minute check
```

The public five minutes are a correctness check only, and the output stays on your own computer.
The rules forbid choosing between versions or settings with them.

## How it works

Version 3 is what `run.py` uses. Versions 1 and 2 stay in the code as comparisons in `bench.py`.

1. **Facts** (`possession/features.py`) -- plain physical questions answered per frame, in metres:
   is a player *controlling* the ball (within a reach that grows with running speed, clearly nearer
   than any opponent, and moving with it rather than past it)? Which team is merely *nearest*? Is
   the ball over a line (Law 9)? Who touched it last and next, did the same team keep it, who was
   there when a long ball came down, and did both teams change direction together?
2. **Two trained models** (`possession/model.py`), both trained on external data only. The team
   model reads those facts and the players' shape to judge who has the ball, including when the
   ball is hidden. The dead-ball model reads the players only, never the ball. Player speed is
   always worked out from positions, so the program does not depend on how a data provider
   computes it. Each model is really three, trained on differently roughened copies of the data,
   with their answers averaged: this cut fake possessions on the Bundesliga and Metrica tests
   without losing real ones.
3. **Restart clues** -- Smart Tagging's set-piece tags vote for a dead ball before each tagged
   restart, and so do the team's restart rules: a ball still for a second, or held at the touchline.
4. **Possessions** (`possession/decode.py`) -- the Viterbi algorithm finds the sequence of states
   with the most support, paying a fixed price for each change of state, with every dead ball at
   least 2 seconds long. This is what turns flickering frames into whole possessions, which is
   what the ranking metric counts.

## Where the settings come from

Every value is in `possession/config.py`, with its reason. No setting and no model comes from the
competition match, its video or its ground truth, as the organisers' clarification of 4 November
2026 requires:

- **The Laws of the Game:** the ball-out rule.
- **External data:** the seven matches of the DFL open dataset. Five chose the settings
  (`calibrate.py`, `tune_v2.py`) with two held out (`bench.py --test`); the live model is then
  trained on all seven, and saved in `models/`, so the submission regenerates byte for byte.
  Three other sources are used only to test it, never to train or tune it (see below).
- **The team's football rules:** the 2-second minimum dead ball, the reach and moving-with-the-ball
  rules, the throw-in and still-ball restart rules.

```bash
python tools/download_dfl.py ../external-data/dfl    # 2.6 GB, verified against published MD5s
python train.py --all                                # the live model: 3 averaged, 7 matches, each also left out once
python tools/accuracy_report.py                      # accuracy on every left-out match and the Clasicos
python bench.py --test                               # held-out DFL matches
python tools/rough_check.py                          # the same matches with rougher tracking
python tools/download_skillcorner.py ../external-data/skillcorner   # 146 MB
python tools/clasico_check.py                        # two Clasicos, real automatic tracking
python tools/download_metrica.py ../external-data/metrica           # Metrica's 3 sample matches
python tools/accuracy_report.py --model possession.pkl --matches MET1,MET2,MET3 --quick
```

Test sets, never trained or tuned on:

| source | matches | tracking | answer key |
|---|---|---|---|
| Metrica Sports sample data | 3 | clean | from Metrica's own events, which people recorded |
| SkillCorner open data | 2 Clasicos | broadcast, automatic, 10 fps | SkillCorner's automatic flag |
| PFF FC 2022 World Cup | 9 | broadcast, automatic | from PFF's own events, which people recorded |

The World Cup data is free from PFF FC on request (a form on their blog post "Get Free Access to
Our Enhanced 2022 World Cup Dataset!"); put it in `../external-data/pff/` and pass
`--matches PFF10502,PFF10512,...` (the list is `external.PFF_MATCHES`). Adding the World Cup or
Metrica matches to training was tried (`train.py --candidate v13`, `v11`): each helped only on its
own source and made the others worse, so neither is used.

The DFL tracking is clean and freezes the ball during stoppages, so `possession/external.py` carries
the ball to the restart spot and degrades the tracking to imitate an automatic tracker.

## Assets used

| asset | used | how |
|---|---|---|
| tactical ATD (players + ball) | yes | all physical facts |
| Smart Tagging | set pieces only | a dead-ball vote before each tagged restart, tuned on imitation tags |
| tactical homography | only in `tools/make_overlay.py` | a private viewing aid, not part of the labels |
| videos, broadcast ATD | no | |

## Known limitations

- The ball is missing on 41% of graded frames. Inside those stretches the team model has only the
  players' shape and the clues either side of the gap.
- External data can only imitate automatic tracking, and DFL's flags are not Metrica's. On two
  Clasicos with real automatic tracking and an automatic answer key, versions 2 and 3 did not beat
  version 1. On real broadcast tracking (the World Cup) the live program does worse than on
  roughened DFL tracking, so the roughening is gentler than the real thing.

## Rules this pipeline keeps

- Every label comes from code. Nothing is labelled or corrected by hand.
- All training and tuning uses external data only.
- No human-produced data about the competition match is used.
- The match data and anything derived from it frame by frame stay out of this repository.
- Our labels are never checked against the video.

## Credits

Tuning data: Deutsche Fussball Liga (DFL), CC BY 4.0. Bassek, Rein, Weber and Memmert (2025), "An
integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12:195,
doi:10.1038/s41597-025-04505-y. Test data: two Clasicos from SkillCorner's first open-data release,
MIT, https://github.com/SkillCorner/opendata (`tools/clasico_check.py`; no setting comes from them).
Test data: Metrica Sports' open sample data, https://github.com/metrica-sports/sample-data. Test
data: PFF FC's 2022 World Cup dataset (now Gradient Sports), free on request from PFF FC.
Metric code: the official More than a Hack 2027 repository, MIT.
