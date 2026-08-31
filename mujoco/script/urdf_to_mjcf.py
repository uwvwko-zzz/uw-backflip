#!/usr/bin/env python3
"""Convert the exact Isaac Gym Go2 URDF into a MuJoCo simulation model.

The stock MuJoCo URDF importer fixes the root link and may infer kilogram-scale
masses for links without inertials.  Isaac Gym instead loads this URDF with a
free base, ``collapse_fixed_joints=True`` and ``density=0.001``.  This converter
keeps the URDF body hierarchy, adds the free base and applies Isaac's fallback
density before adding torque motors and a flat ground plane.
"""

import argparse
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

MESH_FILES = (
    "base_0.obj", "base_1.obj", "base_2.obj", "base_3.obj", "base_4.obj",
    "hip_0.obj", "hip_1.obj",
    "thigh_0.obj", "thigh_1.obj", "thigh_mirror_0.obj", "thigh_mirror_1.obj",
    "calf_0.obj", "calf_1.obj", "calf_mirror_0.obj", "calf_mirror_1.obj",
    "foot.obj",
)


def parser():
    here = Path(__file__).resolve()
    project = here.parents[2]
    result = argparse.ArgumentParser(description="Convert training Go2 URDF to MJCF")
    result.add_argument(
        "--urdf",
        type=Path,
        default=project / "resources/robots/go2/urdf/go2_description.urdf",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=project / "mujoco/go2/go2_isaac.xml",
    )
    return result


def import_preserving_fixed_bodies(urdf_path):
    source = urdf_path.read_text(encoding="utf-8")
    extension = (
        '<mujoco><compiler discardvisual="true" strippath="false" '
        'fusestatic="false"/></mujoco>'
    )
    source, count = re.subn(r"(<robot\b[^>]*>)", r"\1\n  " + extension, source, count=1)
    if count != 1:
        raise ValueError(f"Cannot find <robot> in {urdf_path}")

    with tempfile.TemporaryDirectory(prefix="go2_urdf_to_mjcf_") as tmpdir:
        temporary_urdf = Path(tmpdir) / urdf_path.name
        temporary_xml = Path(tmpdir) / "imported.xml"
        temporary_urdf.write_text(source, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(temporary_urdf))
        mujoco.mj_saveLastXML(str(temporary_xml), model)
        return ET.parse(temporary_xml)


def body_by_name(root, name):
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"Converted model has no body named {name}")


def joint_effort(root, joint_name):
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Converted model has no joint named {joint_name}")
    limits = joint.get("actuatorfrcrange")
    if not limits:
        raise ValueError(f"Joint {joint_name} has no effort limits")
    return limits


def indent_xml(element, level=0):
    """ElementTree.indent replacement for the project's Python 3.8 env."""
    whitespace = "\n" + level * "  "
    child_whitespace = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_whitespace
        for child in element:
            indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_whitespace
        element[-1].tail = whitespace


def insert_visual_geom(body, mesh, material, quat=None):
    attributes = {
        "type": "mesh",
        "mesh": mesh,
        "material": material,
        "group": "2",
        "contype": "0",
        "conaffinity": "0",
        "density": "0",
    }
    if quat:
        attributes["quat"] = quat
    geom = ET.Element("geom", attributes)
    children = list(body)
    child_body_index = next(
        (index for index, child in enumerate(children) if child.tag == "body"),
        len(children),
    )
    body.insert(child_body_index, geom)


def add_visual_meshes(root):
    compiler = root.find("compiler")
    compiler.set("meshdir", "assets")

    asset = ET.Element("asset")
    for name, rgba in (
        ("metal", ".9 .95 .95 1"),
        ("black", "0 0 0 1"),
        ("white", "1 1 1 1"),
        ("gray", "0.671705 0.692426 0.774270 1"),
    ):
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})
    for mesh_file in MESH_FILES:
        ET.SubElement(asset, "mesh", {"name": Path(mesh_file).stem, "file": mesh_file})
    worldbody = root.find("worldbody")
    root.insert(list(root).index(worldbody), asset)

    base = body_by_name(root, "base")
    for mesh, material in (
        ("base_0", "black"), ("base_1", "black"), ("base_2", "black"),
        ("base_3", "white"), ("base_4", "gray"),
    ):
        insert_visual_geom(base, mesh, material)

    hip_quaternions = {
        "FL": None,
        "FR": "4.63268e-05 1 0 0",
        "RL": "4.63268e-05 0 1 0",
        "RR": "2.14617e-09 4.63268e-05 4.63268e-05 -1",
    }
    for leg in ("FL", "FR", "RL", "RR"):
        hip = body_by_name(root, f"{leg}_hip")
        insert_visual_geom(hip, "hip_0", "metal", hip_quaternions[leg])
        insert_visual_geom(hip, "hip_1", "gray", hip_quaternions[leg])

        mirrored = leg in ("FR", "RR")
        thigh = body_by_name(root, f"{leg}_thigh")
        insert_visual_geom(thigh, "thigh_mirror_0" if mirrored else "thigh_0", "metal")
        insert_visual_geom(thigh, "thigh_mirror_1" if mirrored else "thigh_1", "gray")

        calf = body_by_name(root, f"{leg}_calf")
        insert_visual_geom(calf, "calf_mirror_0" if mirrored else "calf_0", "gray")
        insert_visual_geom(calf, "calf_mirror_1" if mirrored else "calf_1", "black")

        foot = body_by_name(root, f"{leg}_foot")
        insert_visual_geom(foot, "foot", "black")


def parse_vector(text, length, default):
    if not text:
        return np.asarray(default, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.size != length:
        raise ValueError(f"Expected {length} values, got {text!r}")
    return value


def quaternion_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def rotate_vector(quaternion, vector):
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(result, vector, quaternion)
    return result


def format_vector(vector):
    return " ".join(f"{value:.10g}" for value in vector)


def collapse_body_into_parent(parent, body):
    body_pos = parse_vector(body.get("pos"), 3, (0, 0, 0))
    body_quat = parse_vector(body.get("quat"), 4, (1, 0, 0, 0))
    for child in list(body):
        if child.tag not in ("geom", "site"):
            continue
        local_pos = parse_vector(child.get("pos"), 3, (0, 0, 0))
        local_quat = parse_vector(child.get("quat"), 4, (1, 0, 0, 0))
        child.set("pos", format_vector(body_pos + rotate_vector(body_quat, local_pos)))
        child.set("quat", format_vector(quaternion_multiply(body_quat, local_quat)))
        body.remove(child)
        children = list(parent)
        index = next((i for i, item in enumerate(children) if item.tag == "body"), len(children))
        parent.insert(index, child)
    parent.remove(body)


def match_isaac_body_collapse(root):
    """Match Isaac's collapse_fixed_joints while preserving dont_collapse links."""
    preserved = {"base", "Head_upper", "Head_lower"}
    for leg in ("FL", "FR", "RL", "RR"):
        preserved.update(
            f"{leg}_{part}" for part in ("hip", "thigh", "calf", "foot")
        )

    def visit(parent):
        for child in list(parent):
            if child.tag != "body":
                continue
            visit(child)
            if child.get("name") not in preserved:
                collapse_body_into_parent(parent, child)

    visit(root.find("worldbody"))


def ensure_visual_assets(output):
    destination = output.parent / "assets"
    source = output.parents[2].parent / "mujoco_menagerie/unitree_go2/assets"
    missing = [name for name in MESH_FILES if not (destination / name).is_file()]
    if not missing:
        return destination
    if not source.is_dir():
        raise FileNotFoundError(
            "Go2 OBJ visual assets are missing. Expected source directory: " + str(source)
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name in missing:
        source_file = source / name
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing visual mesh: {source_file}")
        shutil.copy2(source_file, destination / name)
    return destination


def configure_mjcf(tree, source_urdf):
    root = tree.getroot()
    root.set("model", "go2_isaac_urdf")

    compiler = root.find("compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    compiler.set("fusestatic", "false")

    option = ET.Element(
        "option",
        {
            "timestep": "0.005",
            "integrator": "Euler",
            "solver": "Newton",
            "cone": "pyramidal",
            "iterations": "100",
            "ls_iterations": "50",
        },
    )
    root.insert(list(root).index(compiler) + 1, option)

    visual = ET.Element("visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
    )
    ET.SubElement(visual, "global", {"azimuth": "-130", "elevation": "-20"})
    root.insert(list(root).index(option) + 1, visual)

    worldbody = root.find("worldbody")
    floor = ET.Element(
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            "friction": "1 0 0",
            "condim": "3",
            # Calibrated against the deterministic Isaac trajectory. Keeping
            # equal priority preserves normal parameter mixing and avoids the
            # unstable, overly rigid floor response.
            "priority": "0",
            "solref": "0.02 1",
            "solimp": "0.82 0.95 0.001",
            "rgba": "0.18 0.24 0.30 1",
        },
    )
    worldbody.insert(0, floor)
    worldbody.insert(1, ET.Element("light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"}))

    base = body_by_name(root, "base")
    base.set("pos", "0 0 0.32")
    inertial_index = list(base).index(base.find("inertial"))
    base.insert(inertial_index + 1, ET.Element("freejoint", {"name": "floating_base"}))

    # Isaac uses density=0.001 only when the URDF link has no inertial.  Explicit
    # inertials remain authoritative, while tiny fallback masses keep MuJoCo's
    # fixed child bodies valid without adding Menagerie's ~0.19 kg.
    for geom in root.findall(".//geom"):
        if geom is floor:
            continue
        # Matches Isaac asset.replace_cylinder_with_capsule=True.
        if geom.get("type") == "cylinder":
            geom.set("type", "capsule")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("condim", "3")
        # MuJoCo Viewer shows groups 0-2 by default.  Keep collision shapes in
        # group 3 so they do not cover the OBJ shell; this is visual-only.
        geom.set("group", "3")
        geom.set("friction", "1 0 0")
        geom.set("margin", "0.001")
        geom.set("density", "0.001")
        geom.set("rgba", "0.55 0.60 0.68 1")

    for foot_name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        for geom in body_by_name(root, foot_name).findall("geom"):
            geom.set("rgba", "0.08 0.08 0.08 1")

    match_isaac_body_collapse(root)
    add_visual_meshes(root)

    actuator = ET.SubElement(root, "actuator")
    for joint_name in JOINT_NAMES:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": joint_name[:-6] if joint_name.endswith("_joint") else joint_name,
                "joint": joint_name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": joint_effort(root, joint_name),
            },
        )

    root.insert(0, ET.Comment(f" Generated from {source_urdf} by mujoco/script/urdf_to_mjcf.py "))
    indent_xml(root)
    return tree


def validate(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    expected = set(JOINT_NAMES)
    actual = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    missing = expected - actual
    if missing:
        raise ValueError(f"MJCF missing policy joints: {sorted(missing)}")
    if model.nu != 12:
        raise ValueError(f"Expected 12 actuators, got {model.nu}")
    if model.nq != 19 or model.nv != 18:
        raise ValueError(f"Expected floating Go2 nq/nv=19/18, got {model.nq}/{model.nv}")
    if model.nbody != 20:
        raise ValueError(f"Expected world + 19 Isaac bodies, got {model.nbody}")
    return model


def main():
    args = parser().parse_args()
    urdf = args.urdf.expanduser().resolve()
    output = args.output.expanduser().resolve()
    visual_assets = ensure_visual_assets(output)
    tree = configure_mjcf(import_preserving_fixed_bodies(urdf), urdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    model = validate(output)
    print(f"Wrote       : {output}")
    collision_count = int(((model.geom_contype != 0) | (model.geom_conaffinity != 0)).sum()) - 1
    visual_count = int(((model.geom_contype == 0) & (model.geom_conaffinity == 0)).sum())
    print(f"Bodies      : {model.nbody - 1} robot")
    print(f"Geoms       : {collision_count} collision (+ floor), {visual_count} visual")
    print(f"nq/nv/nu   : {model.nq}/{model.nv}/{model.nu}")
    print(f"Total mass : {model.body_mass.sum():.9f} kg")
    print(f"Visual OBJ : {visual_assets}")


if __name__ == "__main__":
    main()
