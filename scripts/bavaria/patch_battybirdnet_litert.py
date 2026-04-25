"""Patch BattyBirdNET-Analyzer model.py voor ai-edge-litert 2.x compat.

Probleem
========

In ``ai-edge-litert`` 2.x mag ``Interpreter.get_tensor()`` niet meer
gebruikt worden op intermediate tensors. BattyBirdNET-Analyzer haalt
in zijn custom-classifier flow de embedding op via
``OUTPUT_LAYER_INDEX - 1`` (een interne ``GLOBAL_AVG_POOL/Mean``
tensor), wat met de nieuwe runtime crasht met::

    ValueError: Tensor data is null. Run allocate_tensors() first

Resultaat: ``bat_ident.py --area Bavaria`` (en USA-EAST) faalt op
elke WAV. Op emsn-sonar bleef Bavaria daardoor sinds 18 april 2026
voortdurend ``analyzer_failed`` op alle ~5000 verwerkte opnames.

Fix
===

Bij het laden van het BirdNET basis-model in ``loadModel(False)``
(de embeddings flow) zetten we ``experimental_preserve_all_tensors``
op de Interpreter init. Dat houdt alle intermediate tensors
beschikbaar voor latere ``get_tensor()`` calls. Memory cost voor
het BirdNET 6K V2.4 model: ~50-100 MB, ruim binnen Pi 4 budget.

Voor de classificatie-pad (``class_output=True``, BatDetect2-stijl
gebruik) blijft alles bij het oude — daar is geen intermediate
tensor lookup nodig.

Deploy
======

Draai dit script eenmalig op elke Pi waar BattyBirdNET-Analyzer
geïnstalleerd is::

    python3 scripts/bavaria/patch_battybirdnet_litert.py

Het script is idempotent: bij een tweede run detecteert het de
patch en doet niets.

Bij een verse ``git clone`` van BattyBirdNET-Analyzer (third-party
repo van rdz-oss) moet dit script opnieuw gedraaid worden. Upstream
heeft de fix nog niet en accepteert mogelijk nooit een PR omdat
``experimental_preserve_all_tensors`` officieel een debug-flag is.
Voor onze use-case is het echter de minst-invasieve oplossing.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_TARGET = Path.home() / "BattyBirdNET-Analyzer" / "model.py"

OLD_BLOCK = (
    "    if cfg.MODEL_PATH.endswith(\".tflite\"):\n"
    "        # Load TFLite model and allocate tensors.\n"
    "        INTERPRETER = tflite.Interpreter(model_path=cfg.MODEL_PATH, num_threads=cfg.TFLITE_THREADS)\n"
    "        INTERPRETER.allocate_tensors()\n"
)

NEW_BLOCK = (
    "    if cfg.MODEL_PATH.endswith(\".tflite\"):\n"
    "        # Load TFLite model and allocate tensors.\n"
    "        # ai-edge-litert 2.x compat: bij embeddings flow (class_output=False)\n"
    "        # halen we een intermediate tensor op (GLOBAL_AVG_POOL/Mean op\n"
    "        # OUTPUT_LAYER_INDEX-1). TFLite blokkeert get_tensor() voor zulke\n"
    "        # tensors tenzij we experimental_preserve_all_tensors=True zetten.\n"
    "        # Dit kost wat extra geheugen (alle intermediate tensors blijven\n"
    "        # beschikbaar) maar is nodig voor de custom classifier pipeline\n"
    "        # (Bavaria, USA-EAST, etc.).\n"
    "        _interp_kwargs = {\n"
    "            \"model_path\": cfg.MODEL_PATH,\n"
    "            \"num_threads\": cfg.TFLITE_THREADS,\n"
    "        }\n"
    "        if not class_output:\n"
    "            _interp_kwargs[\"experimental_preserve_all_tensors\"] = True\n"
    "        INTERPRETER = tflite.Interpreter(**_interp_kwargs)\n"
    "        INTERPRETER.allocate_tensors()\n"
)

MARKER = "experimental_preserve_all_tensors"


def patch(target: Path) -> int:
    """Pas de patch toe op het opgegeven model.py bestand.

    Returns:
        0 bij succes of als patch al toegepast is, 1 bij fout.
    """
    if not target.exists():
        print(f"FOUT: {target} bestaat niet.", file=sys.stderr)
        return 1

    src = target.read_text()
    if MARKER in src:
        print(f"Patch is al toegepast op {target} - geen wijziging.")
        return 0

    if OLD_BLOCK not in src:
        print(
            f"FOUT: kon de te patchen regio niet vinden in {target}.\n"
            "Mogelijk is het bestand al lokaal aangepast of is de\n"
            "BattyBirdNET-Analyzer versie veranderd. Bekijk diff handmatig.",
            file=sys.stderr,
        )
        return 1

    target.write_text(src.replace(OLD_BLOCK, NEW_BLOCK, 1))
    print(f"Patch toegepast op {target}.")
    return 0


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    return patch(target)


if __name__ == "__main__":
    sys.exit(main())
