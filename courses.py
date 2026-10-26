"""Courses: pick the prompt file and glossary of a lecture by name instead of by path.

Files are grouped by the first word of their name:  prompts/ngs_full.txt, prompts/ngs_3_rnaseq.txt and glossary/ngs.txt all
belong to the course "ngs".  `./run.sh ngs` uses ngs_full (or the only file), `./run.sh ngs 3` uses lesson 3.
"""
import glob
import os
import re

GENERAL = ("full", "default", "short", "simple")      # which file to use when no lesson is named, best first


def scan(prompts_dir, glossary_dir):
    """{course: {"prompts": {label: path}, "glossaries": [path]}}; a label is a lesson number ("3"), "full", "short", ... or "default"."""
    courses = {}
    for path in sorted(glob.glob(os.path.join(prompts_dir, "*.txt"))):
        parts = os.path.splitext(os.path.basename(path))[0].split("_")
        label = parts[1] if len(parts) > 1 and parts[1].isdigit() else (parts[-1] if len(parts) > 1 and parts[-1] in GENERAL else "default")
        courses.setdefault(parts[0], {"prompts": {}, "glossaries": []})["prompts"].setdefault(label, path)
    for path in sorted(glob.glob(os.path.join(glossary_dir, "*.txt"))):
        key = os.path.splitext(os.path.basename(path))[0].split("_")[0]
        courses.setdefault(key, {"prompts": {}, "glossaries": []})["glossaries"].append(path)
    return courses


def resolve(name, courses):
    """The course called `name`, or the only one whose name starts with it. Returns (course, [candidates])."""
    name = name.lower()
    if name in courses:
        return name, [name]
    matches = sorted(k for k in courses if k.startswith(name))
    return (matches[0] if len(matches) == 1 else None), matches


def pick_prompt(course, lesson=None):
    """Path of the prompt file for a course (and lesson), or None."""
    prompts = course["prompts"]
    if lesson is not None:
        return prompts.get(str(lesson))
    for label in GENERAL:
        if label in prompts:
            return prompts[label]
    lessons = sorted((k for k in prompts if k.isdigit()), key=int)
    return prompts[lessons[0]] if lessons else None


def lessons(course):
    return sorted((k for k in course["prompts"] if k.isdigit()), key=int)


def describe(key, course):
    n = len(lessons(course))
    parts = [f"{n} 课提示词" if n else ("有提示词" if course["prompts"] else "无提示词")]
    if course["glossaries"]:
        parts.append("有术语表")
    return f"{key}  ({' · '.join(parts)})"


def read_last(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def write_last(path, key):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(key + "\n")
    except OSError:
        pass


def parse_power(text):
    """("ac" | "battery" | None, charge in percent or None) from the output of macOS `pmset -g batt`."""
    percent = re.search(r"(\d+)%", text)
    charge = int(percent.group(1)) if percent else None
    if re.search(r"Battery Power", text):
        return "battery", charge
    if re.search(r"AC Power", text):
        return "ac", charge
    return None, charge


def power_state():
    import subprocess
    try:
        return parse_power(subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3).stdout)
    except (OSError, subprocess.SubprocessError):
        return None, None


def power_source():
    """"ac", "battery" or None (unknown / not a laptop), from macOS pmset."""
    return power_state()[0]


LOW_BATTERY = 30      # percent: below this, on battery, the automatic profile switches to the one without drafts


def auto_profile(source, charge):
    """The profile --profile auto uses: balanced, unless the laptop runs on a nearly empty battery."""
    return "battery" if source == "battery" and charge is not None and charge < LOW_BATTERY else "balanced"
