"""Build the technical document, SuperSubs.pdf, for the submission form.

    pip install -r tools/requirements-report.txt
    python tools/build_technical_document.py        # writes outputs/SuperSubs.pdf

Every figure in it comes from external data, from the commands named beside each table. Nothing
here reads the competition match, its video or its answer key.
"""

import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import config  # noqa: E402

OUTPUT = os.path.join(config.OUTPUT_DIR, "SuperSubs.pdf")
REPO = "https://github.com/0utlawz/SuperSubs"

INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#57606a")
ACCENT = colors.HexColor("#0b4f8a")
RULE = colors.HexColor("#d0d7de")
SHADE = colors.HexColor("#f3f6f9")
HIGHLIGHT = colors.HexColor("#e8f1fb")

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=11.8, textColor=INK,
                      alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.2, leading=10.6, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.2, leading=10.2, spaceAfter=0)
CELL_HEAD = ParagraphStyle("cellhead", parent=CELL, fontName="Helvetica-Bold", textColor=INK)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=12.5, leading=15,
                    textColor=ACCENT, spaceBefore=9, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=10, leading=13,
                    spaceBefore=5, spaceAfter=2.5)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=19, leading=23,
                       textColor=INK, spaceAfter=3)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=10.5, leading=14, textColor=MUTED,
                          spaceAfter=2)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=11, bulletIndent=2, spaceAfter=2.5)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=8.2, leading=10.4,
                      backColor=SHADE, borderPadding=(4, 5, 4, 5), spaceBefore=3, spaceAfter=7)

WIDTH = A4[0] - 36 * mm


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(item, BULLET, bulletText="•") for item in items]


def table(rows, widths, highlight_rows=(), numeric_from=None):
    """A light-ruled table; rows are lists of strings, the first row is the header."""
    data = [[Paragraph(str(cell), CELL_HEAD if r == 0 else CELL) for cell in row]
            for r, row in enumerate(rows)]
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), HIGHLIGHT))
    built = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=1)
    built.setStyle(TableStyle(style))
    return built


def code(lines):
    return Paragraph("<br/>".join(line.replace(" ", "&nbsp;") for line in lines), CODE)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "SuperSubs — technical document, More than a Hack 2027, Phase 1")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, "page {}".format(doc.page))
    canvas.restoreState()


def story():
    s = [
        p("SuperSubs: ball possession from automatic tracking", TITLE),
        p("More than a Hack 2027, Phase 1 — technical document — November 2026", SUBTITLE),
        p("Team SuperSubs: George, Harout and Carl. Code: <link href='{0}' color='#0b4f8a'>{0}</link>"
          .format(REPO), SUBTITLE),
        Spacer(1, 6),

        p("1. In short", H1),
        p("For every frame of FC Barcelona – Athletic Club the program says who has the ball "
          "(<b>H</b> or <b>A</b>) or that it is out of play (<b>D</b>), and writes <b>X</b> only "
          "outside the two graded halves. Two models trained on <b>external matches only</b> score "
          "each frame: one says which team has the ball, the other whether play is stopped. A "
          "decoder then turns those frame scores into whole possessions, because the ranking "
          "counts whole possessions, not frames. No setting and no model comes from the provided "
          "match, its video or its answer key."),
        p("Measured only on matches the models never saw, against answers people recorded or the "
          "provider published:"),
        table([
            ["test set (never trained on)", "matches", "spellF1 ours", "spellF1 baseline*",
             "real possessions found", "fakes per match"],
            ["Bundesliga, DFL (each match left out in turn)", "7", "<b>0.549</b>", "0.202",
             "48.7%", "107"],
            ["Metrica Sports sample data", "3", "<b>0.593</b>", "0.263", "55.2%", "93"],
            ["2022 World Cup, PFF FC (broadcast tracking)", "9", "<b>0.477</b>", "0.336", "52.6%",
             "191"],
            ["Clasicos, SkillCorner (automatic flag as key)", "2", "0.444 / 0.382", "0.383 / 0.398",
             "–", "–"],
        ], (0.36, 0.08, 0.12, 0.14, 0.15, 0.15)),
        p("* Our re-implementation of the organisers' baseline (nearest player to the ball, majority "
          "vote over 4 s), run on exactly the same cases. Bundesliga and Metrica tracking is roughened to look like "
          "automatic tracking (section 4); World Cup and Clasico tracking is real broadcast "
          "tracking, used as it comes. \"Found\" means a true possession matched at IoU 0.5. "
          "Commands: <font face='Courier'>tools/accuracy_report.py</font>, "
          "<font face='Courier'>tools/clasico_check.py</font>.", SMALL),

        p("2. The approach, and why", H1),
        p("Four steps, all in <font face='Courier'>possession/</font>:"),
    ]
    s += bullets([
        "<b>Facts per frame</b> (<font face='Courier'>features.py</font>), in metres and seconds: "
        "is a player <i>controlling</i> the ball, is a team merely <i>nearest</i>, is the ball "
        "over a line, when was the ball last touched and by which side, where did a long ball come "
        "down and who was there, did both teams turn round together. Player speeds are always "
        "worked out from positions, so nothing depends on a provider's own speed channel.",
        "<b>Two learned models</b> (<font face='Courier'>model.py</font>, gradient-boosted trees, "
        "scikit-learn). The <i>team model</i> gives the chance that the home side has the ball, "
        "from the facts above and both teams' shape (where each team stands, how spread out, which "
        "way it is moving), so it still has something to go on when the ball is missing. The "
        "<i>dead-ball model</i> reads the players only, never the ball: a stoppage shows in how "
        "the whole team slows down and regroups. Each model is in fact three, trained on "
        "differently roughened copies of the training data, with their answers averaged.",
        "<b>Votes for a stoppage</b>: the ball seen past a line, the team's restart rules, and "
        "Smart Tagging's set-piece tags (section 5).",
        "<b>A decoder</b> (<font face='Courier'>decode.py</font>, the Viterbi algorithm) finds "
        "the sequence of H / A / D with the most support, paying a fixed price for every change of "
        "state, with every dead ball at least 2 s long.",
    ])
    s += [
        p("<b>Why this shape.</b> The ranking metric matches whole possession spells at IoU "
          "0.3 to 0.7, so a frame-by-frame classifier that flickers between teams is punished "
          "hard even when most frames are right; the price per change is what keeps possessions "
          "whole. The ball is missing on 41% of graded frames, so the team model must read the "
          "players as well as the ball. And because the rules forbid fitting anything on the "
          "provided match, every model is trained on other matches whose tracking we roughen to "
          "look like automatic tracking. We began with hand-weighted votes (version 1, still in "
          "<font face='Courier'>bench.py</font> for comparison); learning the votes from data "
          "did better on the held-out Bundesliga matches (version 2: 0.461 against 0.332 with the "
          "ball hidden), and version 3 better again."),

        KeepTogether([p("3. Assets: used, and tried and dropped", H1), table([
            ["asset", "used?", "how"],
            ["Tactical ATD (players and ball)", "yes", "every fact in section 2"],
            ["Smart Tagging", "set pieces only",
             "a vote for a dead ball over the 15 s before each tagged restart. The phase tags and a "
             "vote for the restarting team added nothing measurable on imitation tags, so they are "
             "not used."],
            ["Tactical homography and video", "no",
             "only in a private viewing aid (<font face='Courier'>tools/make_overlay.py</font>) that "
             "draws the provided tracking and the public answer key, never our labels"],
            ["Broadcast video and ATD", "no", "–"],
        ], (0.27, 0.13, 0.60))]),
        Spacer(1, 5),
        p("Ideas we built, tested on external data and dropped. Our rule for keeping anything: real "
          "possessions found must hold steady while fake ones fall.", BODY),
        table([
            ["tried", "result on held-out external matches", "kept?"],
            ["Averaging three models (version 3)", "fakes fell by 4 to 6 per match, found unchanged",
             "yes"],
            ["The team's expert rules and where players gather, as extra features (judge 10 "
             "frames after a touch, 15 frames on the ball, nobody within 2 m to tackle)",
             "0.546 against 0.540 on the held-out matches: within the noise", "no"],
            ["Who is closing on the ball", "beaten by the contact-chain features (0.509 against "
             "0.540)", "no"],
            ["A change detector lowering the price of a change of team", "a tie", "no"],
            ["Snapping changes of team to the nearest touch", "0.546 either way", "no"],
            ["A 2-second rule for a change of possession", "lower however it was applied (0.506 "
             "to 0.510 against 0.540), even counting only possessions of 2 s or more", "no"],
            ["Metrica's matches added to training", "no gain: Bundesliga 0.546 to 0.542, Metrica "
             "0.589 to 0.590", "no"],
            ["PFF's World Cup matches added to training", "better on the World Cup only; worse on "
             "the Bundesliga (0.546 to 0.514) and Metrica (0.589 to 0.557)", "no"],
        ], (0.48, 0.42, 0.10), highlight_rows=(1,)),

        p("4. Training data", H1),
        p("Only one labelled source trains the models. None of it comes from the provided match."),
        table([
            ["source", "size", "labels: who and how", "use", "licence"],
            ["DFL open dataset, Bundesliga and 2. Bundesliga (Bassek et al. 2025)",
             "7 matches, 11.1 h of play, 1.0 million frames at 25 fps, 3,061 possessions",
             "ball status (in play or not) and team in possession on every frame, supplied by DFL "
             "with its official tracking", "training", "CC BY 4.0"],
            ["Metrica Sports sample data", "3 matches", "from Metrica's events, which people "
             "recorded (rules below)", "test only", "free use, credit Metrica"],
            ["SkillCorner open data", "2 Clasicos, 10 fps broadcast", "SkillCorner's automatic "
             "possession flag", "test only", "MIT"],
            ["PFF FC 2022 World Cup data", "9 matches with initial-touch events, broadcast",
             "from PFF's events, which people recorded", "test only", "free on request from PFF FC"],
            ["The team's own real-life throw-in test", "one measurement", "the team",
             "one setting (6.4 m/s)", "–"],
        ], (0.25, 0.20, 0.27, 0.13, 0.15), highlight_rows=(1,)),
        Spacer(1, 4),
        p("<b>How the DFL matches are used.</b> DFL tracking is clean, and the competition's is "
          "not, so every training match is used in five roughened versions: the ball hidden on "
          "41% of frames in blocks of 3 to 48 s on average (more often during stoppages in some "
          "versions), slowly drifting position errors of 0.3 to 0.8 m, players dropped, players "
          "far from the play hidden as by a camera, team-mates' tracks swapped, and balls in the "
          "air lost. Each version is also seen with the teams' sides swapped, so the model cannot "
          "learn \"home\" or a club. DFL freezes the ball during stoppages, so it is walked to the "
          "restart spot instead. The DFL possession flag counts very brief touches; following the "
          "rules' \"a touch without control does not count\", spells of the other team shorter "
          "than 1 s are merged into the possession around them before training. Settings were "
          "chosen on five matches with two held out; the final models are trained on all seven."),
        p("<b>Answers for the test sets.</b> Metrica: a team has the ball from its pass, carry, "
          "recovery, restart or shot until the other side does one of those; play is dead from "
          "the ball leaving the pitch, a foul suffered, or a shot going out or in, until the next "
          "restart. World Cup: the same from PFF's events (first touch, pass, cross, carry, "
          "clearance, shot or restart), with play dead from PFF's ball-out-of-play event, which "
          "PFF records before every restart, to the restart itself."),

        p("5. The possession rules we implemented", H1),
    ]
    s += bullets([
        "<b>Control.</b> A player controls the ball when it is within reach (1.23 m when standing, "
        "growing with running speed to 3.02 m at a sprint; each is the 90th percentile of carries "
        "in the DFL data), when they are at least 1 m nearer than any opponent, and when the ball "
        "moves with them (relative speed under 1.73 m/s). A <b>touch</b> is a sudden change in "
        "the ball's velocity (over 9 m/s) with a player within 4 m.",
        "<b>Loose balls, passes and long balls.</b> The team that last played the ball keeps "
        "possession while it travels, including in the air; it changes when the other team plays "
        "it, not when the other team is merely near. Where a long ball comes down, the team model "
        "sees who was there to receive or contest it.",
        "<b>Challenges and deflections.</b> One opposing touch followed by the first team again is "
        "a deflection, not a change of possession (the training answers merge such spells under "
        "1 s). A change of team must be worth its price in the decoder, so a brief duel does not "
        "flip it.",
        "<b>Restarts and stoppages.</b> Play is dead from the ball fully over a line (Law 9: its "
        "centre 0.11 m past the line) or from a stoppage, until the restart. Every dead ball lasts "
        "at least 2 s. Besides the dead-ball model, three clues vote for a stoppage: the ball still "
        "(under 0.5 m/s) for 1 s; the ball held within 1.5 m of the touchline, under 1.5 m/s, for "
        "1 s (a throw-in being taken); and a Smart Tagging set-piece tag in the next 15 s. During a "
        "stoppage no team is given the ball; the restart begins the next possession.",
        "<b>Ball in flight from a throw-in.</b> A ball faster than 6.4 m/s inside the pitch is in "
        "play, from the team's own throw-in test, so a short throw counts as soon as it enters "
        "the pitch.",
        "<b>No answer (X).</b> Only outside the two graded halves. Inside them every frame gets "
        "H, A or D.",
    ])
    s += [
        p("6. Limitations, and what we would fix with more time", H1),
    ]
    s += bullets([
        "<b>Short possessions are missed.</b> About half of the real possessions are found. The "
        "program reports fewer, longer ones than the answers (318 against 437 per Bundesliga "
        "match), and misses most spells under 2 s, which are many in fast transitions.",
        "<b>Real automatic tracking is harder than our roughening.</b> On real broadcast tracking "
        "(World Cup) the program scores 0.477 against 0.549 on roughened Bundesliga tracking, and "
        "reports too many possessions (345 against 293). On the two Clasicos, scored against "
        "SkillCorner's automatic flag, our hand-weighted version 1 did better (0.559 / 0.517) "
        "than the learned version, and the baseline was slightly ahead of it on one of them. The tactical "
        "camera of the provided match sees nearly every player, so it should sit between the two.",
        "<b>A small training set.</b> Seven matches, five with the same home team. Adding World "
        "Cup or Metrica matches to training did not improve the other tests.",
        "<b>Quick restarts</b> taken in under 2 s cannot be called dead, by design.",
        "<b>With more time:</b> labelled matches from the same kind of tactical-camera automatic "
        "tracking; tuning the decoder's prices together with the models; and a model of the "
        "ball's path through gaps instead of a straight line.",
    ])
    s += [
        p("7. Rules, use of AI, and an open disclosure", H1),
    ]
    s += bullets([
        "Every label comes from code; nothing is labelled or corrected by hand.",
        "All training and tuning uses external data only, as the organisers' clarification of "
        "4 November 2026 requires. No human-produced data about the provided match is used.",
        "The match data, and anything derived from it frame by frame, are not in the repository.",
        "Our labels have never been checked against the video.",
    ])
    s += [
        p("<b>Use of AI.</b> The code was written with Claude Code, Anthropic's AI coding assistant, "
          "working on our computer under our direction. The football rules, every judgment call "
          "and every choice between versions were ours. Since the clarification, the assistant "
          "has not been given the provided match data or any number computed from it. It read "
          "only the public five-minute answer key in the organisers' repository, to understand "
          "the definitions, and when it ran the program on the match it saw only that the file "
          "was written, that the official format check passed, and that two runs gave identical "
          "files."),
        p("<b>Disclosure.</b> Before the clarification reached us, we explored the provided match "
          "with the assistant. Summary statistics from it (for example how long the ball goes "
          "missing and how still a placed ball is), file names and field names appeared in that "
          "chat, and we ran the official scorer on the public five minutes as a pipeline check. "
          "A few early settings were measured on the match then. When the email arrived, all of "
          "them were discarded and measured again on external data, and the training and the final "
          "choice of model were done on external matches only. The general design was still "
          "partly shaped by what we saw before the email. We state this openly so the organisers "
          "can judge it."),

        KeepTogether([p("8. Reproduction", H1),
        p("Python 3.9; package versions are pinned in "
          "<font face='Courier'>requirements.txt</font>, and the trained model is in the "
          "repository, so the submission regenerates byte for byte. The official repository is "
          "cloned next to ours with the match data in its <font face='Courier'>data/</font> "
          "folder."),
        code([
            "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
            ".venv/bin/python run.py                  # -> outputs/SuperSubs.csv",
            "",
            "# retrain from scratch on the DFL open data (about 30 minutes)",
            ".venv/bin/python tools/download_dfl.py ../external-data/dfl",
            ".venv/bin/python train.py --all          # three models averaged, 7 matches",
            ".venv/bin/python tools/accuracy_report.py   # every match left out once",
        ])]),

        p("Credits", H2),
        p("Training data: Deutsche Fussball Liga (DFL), CC BY 4.0; Bassek, Rein, Weber and Memmert "
          "(2025), \"An integrated dataset of spatiotemporal and event data in elite soccer\", "
          "Scientific Data 12:195, doi:10.1038/s41597-025-04505-y. Test data: Metrica Sports' "
          "open sample data (github.com/metrica-sports/sample-data); SkillCorner open data, MIT "
          "(github.com/SkillCorner/opendata); PFF FC's 2022 World Cup dataset (now Gradient "
          "Sports). Metric code: the official More than a Hack 2027 repository, MIT. Models: "
          "scikit-learn.", SMALL),
    ]
    return s


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=13 * mm, bottomMargin=15 * mm,
                            title="SuperSubs — technical document", author="SuperSubs",
                            subject="More than a Hack 2027, Phase 1: ball possession")
    doc.build(story(), onFirstPage=footer, onLaterPages=footer)
    print("wrote {}".format(os.path.relpath(OUTPUT, ROOT)))


if __name__ == "__main__":
    main()
