from pathlib import Path
import json
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parent


def source_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def build_notebook():
    cells = [markdown_cell("""# ACDC Cardiac MRI Segmentation and Diagnosis

This notebook is the consolidated reproducibility deliverable for Group 26. It contains the source code used for preprocessing, the classical baseline, U-Net experiments, diagnosis experiments, and final test evaluation.

The long training sections are guarded by `RUN_FULL_EXPERIMENTS = False` so the notebook can be inspected and checked without accidentally starting multi-minute training runs. Set it to `True` after configuring the data path to reproduce the experiments. The result tables below load the verified JSON outputs already produced by the runs.

The experiment design is patient-level: 80 training patients, 20 validation patients, and 50 untouched test patients. Do not split individual slices randomly across these sets.
"""),
        markdown_cell("## 1. Configuration and verified outputs"),
        source_cell("""from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd()
PROJECT = ROOT
DATA_ROOT = PROJECT / 'cmps261-work/data/acdc/extracted/database'
RUN_FULL_EXPERIMENTS = False

with open(PROJECT / 'results/experiments.json') as f:
    experiments = json.load(f)

print('Project folder:', PROJECT.name)
print('Data root:', DATA_ROOT.relative_to(PROJECT), '(exists)' if DATA_ROOT.exists() else '(download ACDC here)')
print('Verified U-Net choice:', experiments['unet_chosen'])
print('Verified U-Net test macro-F1:', experiments['test_segmentation']['U-Net']['test']['macro_f1'])
"""),
        markdown_cell("## 2. Data loading and preprocessing source"),
    ]

    files = [
        ("acdc_forest_baseline.py", "## 3. Classical baseline source\n\nThis section contains the forest preprocessing, sampling, tuning, validation, and reporting code."),
        ("acdc_unet_pilot.py", "## 4. Initial U-Net source\n\nThis section contains the initial U-Net training and validation pipeline."),
        ("acdc_experiments.py", "## 5. U-Net variants, diagnosis, and final test source\n\nThis section contains the controlled model comparison, diagnosis models, and the single final test evaluation."),
        ("acdc_diagnosis_baseline.py", "## 6. Diagnosis baseline source\n\nThis section contains the expert-outline diagnosis baseline and saved classical diagnosis model."),
    ]
    for filename, title in files:
        cells.append(markdown_cell(title))
        source = (ROOT / filename).read_text()
        guarded = "# Set RUN_FULL_EXPERIMENTS = True above to execute this long-running section.\nif RUN_FULL_EXPERIMENTS:\n" + "\n".join("    " + line if line else "" for line in source.splitlines()) + "\n"
        cells.append(source_cell(guarded))

    cells += [
        markdown_cell("## 7. Verified result tables\n\nThese tables are loaded from the saved experiment outputs. They are not recomputed from the test set during report writing."),
        source_cell("""forest = experiments['test_segmentation']['random forest']['test']
unet = experiments['test_segmentation']['U-Net']['test']
summary = pd.DataFrame([
    {'model': 'Random forest', 'macro_F1': forest['macro_f1'], 'pixel_accuracy': forest['pixel_accuracy'], 'seconds_per_slice': experiments['test_segmentation']['random forest']['seconds_per_slice']},
    {'model': 'U-Net', 'macro_F1': unet['macro_f1'], 'pixel_accuracy': unet['pixel_accuracy'], 'seconds_per_slice': experiments['test_segmentation']['U-Net']['seconds_per_slice']},
])
display(summary)

diagnosis = experiments['test_diagnosis']
display(pd.DataFrame([
    {'outline_source': 'Expert outlines', 'logistic_regression_accuracy': diagnosis['accuracy']['logistic regression C=0.1']['expert outlines']},
    {'outline_source': 'U-Net outlines', 'logistic_regression_accuracy': diagnosis['accuracy']['logistic regression C=0.1']['U-Net outlines']},
    {'outline_source': 'Forest outlines', 'logistic_regression_accuracy': diagnosis['accuracy']['logistic regression C=0.1']['forest outlines']},
]))
"""),
        markdown_cell("## 8. Reproducibility notes\n\nThe original runs used the bundled `cmps261` environment, PyTorch with MPS where available, scikit-learn, nibabel, SciPy, matplotlib, joblib, and NumPy. The ACDC data and saved model files were kept outside iCloud during experimentation because of size and runtime constraints. Before submission, update the paths or add a README explaining how the data and model files are obtained. The final test set must remain untouched during any new tuning."),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "Group 26 Final Project Notebook.ipynb").write_text(json.dumps(nb, indent=1))


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def add_table(doc, rows):
    table = doc.add_table(rows=1, cols=len(rows[0]))
    table.style = 'Table Grid'
    for i, value in enumerate(rows[0]):
        cell = table.rows[0].cells[i]
        cell.text = value
        set_cell_shading(cell, '1F4E78')
        for run in cell.paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
    for row in rows[1:]:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            clean = value.replace('**', '')
            cells[i].text = clean
            if value.startswith('**') and value.endswith('**'):
                for run in cells[i].paragraphs[0].runs:
                    run.bold = True
    doc.add_paragraph()


def build_docx():
    text = (ROOT / 'Final Report.md').read_text()
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    styles = doc.styles
    styles['Normal'].font.name = 'Aptos'
    styles['Normal'].font.size = Pt(10.5)
    for name, size in [('Title', 20), ('Heading 1', 15), ('Heading 2', 12)]:
        styles[name].font.name = 'Aptos Display'
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = RGBColor(0, 0, 0)

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(18)
            r = p.add_run(line[2:].strip())
            r.font.name = 'Aptos Display'
            r.font.size = Pt(20)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0, 0, 0)
        elif line.startswith('## '):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.startswith('### '):
            doc.add_heading(line[4:].strip(), level=2)
        elif line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                if not re.match(r'^\|\s*:?-+', lines[i]):
                    block.append([x.strip() for x in lines[i].strip('|').split('|')])
                i += 1
            add_table(doc, block)
            continue
        elif line.startswith('- '):
            doc.add_paragraph(line[2:].strip(), style='List Bullet')
        elif re.match(r'^\d+\. ', line):
            doc.add_paragraph(re.sub(r'^\d+\. ', '', line), style='List Number')
        elif line.strip():
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(5)
            # Basic markdown emphasis for readability.
            parts = re.split(r'(\*\*[^*]+\*\*)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    r = p.add_run(part[2:-2]); r.bold = True
                else:
                    p.add_run(part)
        i += 1

    doc.add_page_break()
    doc.add_heading('Selected figures', level=1)
    for image, caption in [
        ('results/forest_examples.png', 'Random-forest predictions compared with expert outlines.'),
        ('results/test_examples.png', 'Final U-Net predictions on held-out test patients.'),
    ]:
        path = ROOT / image
        if path.exists():
            doc.add_picture(str(path), width=Inches(6.3))
            p = doc.add_paragraph(caption)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].italic = True
    doc.save(ROOT / 'Group 26 Final Report.docx')


if __name__ == '__main__':
    build_notebook()
    build_docx()
    print('Built notebook and final report.')
