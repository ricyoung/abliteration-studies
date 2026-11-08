"""Figure 1 (study workflow). Final design = fig1_candidates/fig1_final.py (design round 2026-11-05: 3 concepts,
2 judges, winner B polished); this wrapper renders it and installs it as paper/figures/fig1_workflow.{pdf,png}."""
import shutil, subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
paper = here.parent.parent / "paper" / "figures"
subprocess.run([sys.executable, str(here / "fig1_candidates" / "fig1_final.py")], check=True, cwd=here / "fig1_candidates")
shutil.copyfile(here / "fig1_candidates" / "fig1_final.pdf", paper / "fig1_workflow.pdf")
shutil.copyfile(paper / "preview" / "fig1_candidates" / "fig1_final.png", paper / "preview" / "fig1_workflow.png")
print("wrote", paper / "fig1_workflow.pdf")
