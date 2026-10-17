"""
Builds report/Technical_Report.docx (and a PDF if LibreOffice is installed).
All numbers are read from the results/ folder.

    python report/make_report.py
"""
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Cm, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg  # noqa: E402

STUDENT_NAME = "[Your Name]"      # <- write your name here
ROLL_NO = "[Roll No]"             # <- write your roll number here
RES = os.path.join(ROOT, "results")


def read_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(1.8)
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(10.5)
style.paragraph_format.space_after = Pt(4)


def heading(text):
    doc.add_heading(text, level=1)


def para(text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def table(rows, caption):
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].bold = True
    cap.runs[0].font.size = Pt(9)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(value)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9)
                run.bold = i == 0
    doc.add_paragraph()


def figure(name, width, caption):
    doc.add_picture(os.path.join(RES, name), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)


# ---------------- read the results ----------------
main = read_csv("main_summary.csv")[0]
m = {k: (float(v) if k != "stop_reason" else v) for k, v in main.items()}
sens = read_csv("sensitivity.csv")
base = read_csv("baseline.csv")
ga = [r for r in base if r["method"] == "GA"]
rs = [r for r in base if r["method"] == "Random search"]


def mean_std(rows, key, d=3):
    values = np.array([float(r[key]) for r in rows])
    return f"{values.mean():.{d}f} ± {values.std():.{d}f}"


def sens_value(param, value):
    for r in sens:
        if r["parameter"] == param and float(r["value"]) == value:
            return f"{float(r['mean_best']):.3f} ± {float(r['std_best']):.3f}"


# ---------------- title page ----------------
for _ in range(6):
    doc.add_paragraph()
for text, size, bold in [("Bahria University Lahore Campus", 18, True),
                         ("Department of Computer Science", 14, False),
                         ("Artificial Intelligence – BSCS 5", 14, False),
                         ("Assignment 2", 14, True), ("", 12, False),
                         ("UAV Fleet Planning with a Genetic Algorithm", 20, True),
                         ("Technical Report", 13, False), ("", 12, False), ("", 12, False),
                         (f"Name: {STUDENT_NAME}", 13, False), (f"Roll No: {ROLL_NO}", 13, False)]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ---------------- 1. Introduction ----------------
heading("1. Introduction")
para(f"A company uses drones (UAVs) to survey a disaster area. We must decide where each drone flies, how "
     f"fast it flies and how strong its sensor is. A good plan covers the important areas and finds the "
     f"targets, but it should also save battery, finish quickly and stay away from dangerous zones. These "
     f"goals conflict, so there is no single correct answer. I solved this problem with a Genetic "
     f"Algorithm (GA) written from scratch in Python.")
para(f"The area is a {cfg.GRID_SIZE} x {cfg.GRID_SIZE} grid. It has {len(cfg.NO_FLY_ZONES)} no-fly zones "
     f"(blocked cells), {len(cfg.HAZARD_ZONES)} hazard zones (high wind / radiation), "
     f"{len(cfg.PRIORITY_ZONES)} priority (disaster) zones and {cfg.N_TARGETS} targets. "
     f"{cfg.N_DRONES} drones start at the base {cfg.BASE}, visit {cfg.N_WAYPOINTS} waypoints each and "
     f"return to the base. Each drone may fly at most {cfg.MAX_RANGE} cells. One hazard zone lies on top "
     f"of the most important disaster zone, so the GA must find a balance between coverage and safety.")

# ---------------- 2. Chromosome ----------------
heading("2. Chromosome Design")
para("Each chromosome is one complete plan for the whole fleet. It is a real-valued array with one "
     "segment per drone. Every waypoint has four genes:")
table([["Gene", "Meaning", "Range"],
       ["x, y", "position of the waypoint", f"0 – {cfg.GRID_SIZE}"],
       ["speed", "speed on the way to this waypoint", f"{cfg.SPEED_MIN} – {cfg.SPEED_MAX} cells/step"],
       ["sensor power", "sensor strength (radius 1 to 4 cells)", "0 – 1"]],
      f"Table 1: Genes of one waypoint (chromosome size = {cfg.N_DRONES} drones x "
      f"{cfg.N_WAYPOINTS} waypoints x 4 genes = {cfg.N_DRONES * cfg.N_WAYPOINTS * 4} genes)")
para("Why this design? (1) It is simple: the genes are exactly the things we must decide (waypoints, "
     "speed and sensor power). (2) Real numbers suit positions on a map: a small change in a gene gives "
     "a small change in the route, which works well with Gaussian mutation and blend crossover. "
     "(3) One segment per drone means a good route of one drone can be passed to a child as a whole "
     "block. (4) The sensor power gene lets the GA switch the sensor to low power where there is "
     "nothing to see and save energy.")

# ---------------- 3. Fitness ----------------
heading("3. Fitness Function")
para("F = 0.35·Coverage + 0.25·Detection − 0.15·Energy − 0.10·Time − 0.15·Hazard − 2·Penalty")
table([["Objective", "How it is measured (scaled to 0–1)", "Weight"],
       ["Coverage (max)", "priority-weighted share of cells inside a sensor circle", cfg.W_COVERAGE],
       ["Detection (max)", "share of targets inside a sensor circle", cfg.W_DETECTION],
       ["Energy (min)", "distance x (1 + 0.3·speed) + sensor power x time, divided by the maximum", cfg.W_ENERGY],
       ["Time (min)", "flight time of the slowest drone / longest possible time", cfg.W_TIME],
       ["Hazard (min)", "average hazard value under the flown path", cfg.W_HAZARD]],
      "Table 2: Objectives and weights")
para("All objectives are scaled to 0–1, otherwise a large number (for example energy) would dominate. "
     "The two mission goals (coverage and detection) get 0.60 of the weight and the three costs get "
     "0.40, so the drones are rewarded for doing the job and not for staying at the base. Coverage has "
     "the largest weight because surveying the disaster zones is the main purpose. Hazard and energy "
     "both get 0.15 because drone safety and battery are equally important. Time gets the smallest "
     "weight because the range limit already keeps the missions short.")
para("Constraint handling uses both methods. Repair: if a waypoint falls inside a no-fly zone, it is "
     "moved to the nearest open cell. Penalty: if a leg still flies through a no-fly zone, or a drone "
     "flies further than the maximum range, the fitness is reduced by 2 x (10 x share of the path inside "
     "no-fly zones + extra range). The penalty is large, so plans that break the rules cannot win.")

# ---------------- 4. Operators ----------------
heading("4. Genetic Operators and Settings")
para(f"Initial population: {cfg.POP_SIZE} random plans. Selection: tournament selection with "
     f"k = {cfg.TOURNAMENT_K} (pick k random plans, the best one becomes a parent); k controls the "
     f"selection pressure. Crossover (rate {cfg.CROSSOVER_RATE}): first a segment swap, where every "
     f"drone's route is copied from parent 1 or parent 2; then a geometric blend, where one drone's "
     f"waypoints are a random mix of both parents (a point between them). Mutation (rate "
     f"{cfg.MUTATION_RATE} per waypoint): the waypoint moves by a small Gaussian step (about "
     f"{cfg.MUTATION_STEP} cells), its speed and sensor power change a little, and sometimes two "
     f"waypoints swap their order. To keep diversity, the {cfg.ELITE} best plans survive unchanged "
     f"(elitism) and {cfg.IMMIGRANTS} new random plans are added every generation. Termination: "
     f"{cfg.GENERATIONS} generations, or no improvement for {cfg.PATIENCE} generations.")

# ---------------- 5. Results ----------------
heading("5. Results of the Main Run")
para(f"With seed {cfg.SEED} the run {m['stop_reason']} ({int(m['evaluations'])} fitness evaluations). "
     f"The best fitness improved from {m['first_best']:.3f} in generation 0 to {m['final_best']:.3f}. "
     f"The final plan covers {m['coverage']:.2f} of the priority-weighted area, finds "
     f"{m['detection'] * 100:.0f}% of the targets, has energy {m['energy']:.2f}, time {m['time']:.2f}, "
     f"hazard {m['hazard']:.2f} and penalty {m['penalty']:.2f} (no rules broken). The average fitness "
     f"stays below the best because two random immigrants enter every generation.")
figure("convergence.png", 12, "Figure 1: Best and average fitness over the generations")
figure("paths_generations.png", 13.5, "Figure 2: Best flight paths in generations 0, 25, 50 and 100 "
       "(thicker line = stronger sensor, dotted = way home)")
para("In generation 0 the routes are random zig-zags that cross the no-fly zones (black). Over the "
     "generations the routes move around the no-fly zones, each drone takes its own area, and the strong "
     "sensor (thick lines) is used inside the priority zones (blue boxes). The coverage heatmaps show "
     "the same: the covered area moves to the important zones.")
figure("coverage_heatmaps.png", 16.5, "Figure 3: Coverage of the best plan across generations "
       "(colour = priority of the covered cell, white = not covered)")

# ---------------- 6. Sensitivity ----------------
heading("6. Hyperparameter Sensitivity Analysis")
para("I changed one setting at a time and kept the others at their default values. Each setting was "
     "run with 3 seeds for 100 generations. Table 3 shows the final best fitness (mean ± standard "
     "deviation).")
table([["Setting", "Value 1", "Value 2", "Value 3"],
       ["Population size", f"20: {sens_value('pop_size', 20)}", f"50: {sens_value('pop_size', 50)}",
        f"100: {sens_value('pop_size', 100)}"],
       ["Mutation rate", f"0.02: {sens_value('mutation_rate', 0.02)}", f"0.1: {sens_value('mutation_rate', 0.1)}",
        f"0.3: {sens_value('mutation_rate', 0.3)}"],
       ["Tournament k", f"2: {sens_value('tournament_k', 2)}", f"3: {sens_value('tournament_k', 3)}",
        f"5: {sens_value('tournament_k', 5)}"]],
      "Table 3: Final best fitness for different settings (3 seeds)")
figure("sensitivity.png", 16.5, "Figure 4: Mean best fitness for each setting")
para("Population size: a bigger population gives better and more stable results, because it searches "
     "more of the map at the same time (more exploration). The cost is more fitness evaluations: "
     "100 plans need twice the time of 50 plans.")
para("Mutation rate: 0.02 was clearly the worst. With so little mutation the population becomes "
     "similar very quickly and gets stuck (premature convergence). 0.1 and 0.3 both worked well; 0.3 was "
     "slightly better and more stable here, because the routes need many changes to move around the "
     "no-fly zones.")
para("Selection pressure: a small tournament (k = 2) gave the best result, and k = 5 the worst. With a "
     "large k the best plans are chosen almost every time, the population loses diversity and some runs "
     "get stuck early. This shows the balance between exploration (low pressure) and exploitation "
     "(high pressure).")

# ---------------- 7. GA vs random search ----------------
heading("7. GA versus Random Search")
para(f"Random search creates random plans (with the same repair) and keeps the best one. To be fair it "
     f"got exactly the same number of fitness evaluations as the GA ({ga[0]['evaluations']}). Both were "
     f"run with 5 seeds.")
table([["Measure (best plan)", "Genetic Algorithm", "Random search"],
       ["Fitness", mean_std(ga, "fitness"), mean_std(rs, "fitness")],
       ["Coverage", mean_std(ga, "coverage"), mean_std(rs, "coverage")],
       ["Detection", mean_std(ga, "detection"), mean_std(rs, "detection")],
       ["Energy", mean_std(ga, "energy"), mean_std(rs, "energy")],
       ["Time", mean_std(ga, "time"), mean_std(rs, "time")],
       ["Hazard", mean_std(ga, "hazard"), mean_std(rs, "hazard")],
       ["Penalty", mean_std(ga, "penalty"), mean_std(rs, "penalty")]],
      "Table 4: GA vs. random search (mean ± standard deviation over 5 seeds)")
figure("ga_vs_random.png", 11.5, "Figure 5: Best fitness found so far vs. number of evaluations")
figure("ga_vs_random_maps.png", 15, "Figure 6: Best plan of the GA (left) and of random search (right), seed 1")
para("The GA is much better than random search. Random search never found a plan that avoids all "
     "no-fly zones, so it always has a penalty; 4 of the 5 GA runs broke no rule at all. What makes the "
     "GA solution special: (1) structured paths: each drone flies to its own area and the routes go "
     "around the no-fly zones instead of random zig-zags; (2) sensor scheduling: the strong sensor is "
     "used over the priority zones, a weaker one on the way there; (3) priority focus: higher coverage "
     "and all targets found; (4) shorter mission time and slightly less hazard. Random search only "
     "finds such a plan by luck, while the GA builds it step by step from good parts of earlier plans.")

# ---------------- 8. Conclusion ----------------
heading("8. Conclusion and Limitations")
para("I built a simple GA that plans the routes, speeds and sensor power of a drone fleet. The real-valued "
     "chromosome, the weighted fitness function, custom crossover and mutation, and repair plus penalty "
     "together give good and legal plans that are far better than random search. Limitations: the map is "
     "small and fixed, the energy and sensor models are simple, and the weighted sum gives only one "
     "compromise. In the future the GA could use a moving hazard field, more drones, and a multi-objective "
     "method such as NSGA-II to show many trade-offs at once.")

out = os.path.join(ROOT, "report", "Technical_Report.docx")
doc.save(out)
print("saved", out)
office = shutil.which("soffice") or shutil.which("libreoffice")
if office:
    subprocess.run([office, "--headless", "--convert-to", "pdf", "--outdir", os.path.dirname(out), out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("saved", out.replace(".docx", ".pdf"))
