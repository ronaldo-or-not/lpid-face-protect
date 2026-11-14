#!/usr/bin/env python3
"""LPID -- protect face photos against unauthorized recognition training.

Takes the photos of one or more people and writes out protected copies: each
output photo is visually the same as its input, but a model trained on the
protected copies fails to recognize the person in ordinary photos.

The perturbation is confined to the face box the scraper would crop, and it is
optimized *through* a differentiable model of that crop+resize, so it survives
the extraction step instead of being wiped out by the resize.

Algorithm and hyperparameters follow generate_lpid_v2 in _build_expABC.py:
  inner  20-step sign-PGD descent on the error-minimizing loss, eps 8/255,
         step 0.8/255, reprojected so the published photo stays in [0,1]
  outer  ResNet-18 surrogate, SGD lr 0.025 / momentum 0.9 / wd 5e-4,
         cosine annealing, up to 60 epochs, early stop at 0.99 train accuracy

IMPORTANT -- this method needs an ALBUM, not a single photo. The optimization
trains a surrogate classifier, so with one identity the cross-entropy loss is
identically zero and nothing is generated. Supply at least two identities:
either several people under --input, or one person plus --refs.

Usage:
    python3 protect.py --input ./my_photos --refs ./reference_ids --output ./protected
    python3 protect.py --self-test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- io

def load_photo(path):
    """Read an image as a float CHW tensor in [0,1]."""
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def save_photo(tensor, path):
    """Write a float CHW tensor in [0,1] as lossless PNG."""
    array = (tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path, format="PNG", optimize=True)


def list_identities(root, label):
    """Map identity name -> list of image paths. Subfolders are identities."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"{label} 경로가 폴더가 아닙니다: {root}")
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    groups = {}
    if subdirs:
        for folder in subdirs:
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if files:
                groups[folder.name] = files
    else:
        files = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            groups[root.name] = files
    if not groups:
        raise SystemExit(f"{label} 폴더에서 이미지를 찾지 못했습니다: {root}")
    return groups


# ------------------------------------------------------------------- face boxes

def square_box(x1, y1, x2, y2, width, height):
    """Largest square box centred on the detection and inside the image."""
    centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = int(round(max(x2 - x1, y2 - y1)))
    side = max(1, min(side, width, height))
    x = max(0, min(int(round(centre_x - side / 2)), width - side))
    y = max(0, min(int(round(centre_y - side / 2)), height - side))
    return x, y, side


def detect_boxes(paths, device, min_face):
    """Run MTCNN once per photo and return {path: (x, y, side)}; misses are skipped."""
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        raise SystemExit(
            "얼굴 검출에 facenet-pytorch 가 필요합니다.\n"
            "  pip install facenet-pytorch\n"
            "또는 --boxes 로 얼굴 영역을 직접 지정하세요.")
    detector = MTCNN(keep_all=True, device=device)
    boxes, skipped = {}, []
    for path in paths:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        found, probability = detector.detect(image)
        if found is None or len(found) == 0:
            skipped.append((path, "얼굴 검출 실패")); continue
        best = int(np.argmax(probability))
        x, y, side = square_box(*found[best], width, height)
        if side < min_face:
            skipped.append((path, f"얼굴이 너무 작음 ({side}px < {min_face}px)")); continue
        boxes[path] = (x, y, side)
    return boxes, skipped


def load_boxes(path, paths, min_face):
    """Read {relative-or-absolute path: [x, y, side]} from a JSON file."""
    raw = json.loads(Path(path).read_text())
    boxes, skipped = {}, []
    for photo in paths:
        entry = None
        for key in (str(photo), photo.name, str(photo.resolve())):
            if key in raw:
                entry = raw[key]; break
        if entry is None:
            skipped.append((photo, "--boxes 에 항목이 없음")); continue
        x, y, side = (int(v) for v in entry)
        if side < min_face:
            skipped.append((photo, f"얼굴이 너무 작음 ({side}px < {min_face}px)")); continue
        boxes[photo] = (x, y, side)
    return boxes, skipped


# ---------------------------------------------------------------- optimization

def new_surrogate(arch, num_classes):
    import torchvision.models as models
    factory = dict(resnet18=models.resnet18, resnet50=models.resnet50,
                   vgg16=models.vgg16_bn, densenet121=models.densenet121)
    if arch not in factory:
        raise SystemExit(f"지원하지 않는 대리 모델입니다: {arch}")
    return factory[arch](num_classes=num_classes)


def extract(faces, deltas, order, target_size):
    """Differentiable model of the scraper's extraction: crop face, resize to T.

    The crop is already the stored face box, so this is the resize; gradients of
    the post-resize loss flow back into each per-photo delta.
    """
    crops = []
    for i in order:
        published = (faces[i] + deltas[i]).clamp(0, 1)
        crops.append(F.interpolate(published[None], (target_size, target_size),
                                   mode="bilinear", align_corners=False))
    return torch.cat(crops, 0)


def generate(faces, labels, num_classes, args, device, log=print):
    """Alternate inner perturbation descent with outer surrogate training."""
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = new_surrogate(args.arch, num_classes).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    noise = [torch.zeros_like(face) for face in faces]        # kept on CPU
    label_tensor = torch.tensor(labels)
    rng = random.Random(args.seed)
    indices = list(range(len(faces)))

    for epoch in range(args.epochs):
        rng.shuffle(indices)
        correct = total = 0
        for start in range(0, len(indices), args.batch_size):
            order = indices[start:start + args.batch_size]
            if len(order) < 2:
                continue                                      # a 1-sample batch teaches nothing
            clean = {i: faces[i].to(device) for i in order}
            delta = {i: noise[i].to(device).requires_grad_(True) for i in order}
            y = label_tensor[order].to(device)

            model.eval()
            for _ in range(args.steps):
                crops = extract(clean, delta, order, args.target_size)
                loss = F.cross_entropy(model((crops - mean) / std), y)
                grads = torch.autograd.grad(loss, [delta[i] for i in order])
                with torch.no_grad():
                    for i, grad in zip(order, grads):
                        step = delta[i].detach() - args.step_size * grad.sign()
                        step = step.clamp(-args.epsilon, args.epsilon)
                        step = (clean[i] + step).clamp(0, 1) - clean[i]   # publishable photo
                        delta[i] = step.requires_grad_(True)
            for i in order:
                noise[i] = delta[i].detach().cpu()

            model.train()
            frozen = {i: noise[i].to(device) for i in order}
            crops = extract(clean, frozen, order, args.target_size)
            optimiser.zero_grad(set_to_none=True)
            logits = model((crops - mean) / std)
            F.cross_entropy(logits, y).backward()
            optimiser.step()
            correct += int((logits.argmax(1) == y).sum()); total += len(order)
        schedule.step()
        accuracy = correct / max(total, 1)
        if epoch < 3 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            log(f"  epoch {epoch + 1}/{args.epochs}  대리 모델 학습 정확도 {accuracy:.3f}")
        if accuracy >= args.stop_acc:
            log(f"  조기 종료: epoch {epoch + 1} 에서 학습 정확도 {accuracy:.3f} 도달")
            break
    return noise


# ----------------------------------------------------------------------- driver

def run(args, device, log=print):
    protect_groups = list_identities(args.input, "--input")
    reference_groups = list_identities(args.refs, "--refs") if args.refs else {}
    overlap = set(protect_groups) & set(reference_groups)
    if overlap:
        raise SystemExit(f"--input 과 --refs 에 같은 이름의 신원이 있습니다: {sorted(overlap)}")

    num_classes = len(protect_groups) + len(reference_groups)
    if num_classes < 2:
        raise SystemExit(
            "신원이 1개뿐입니다. 이 방법은 사진 한 장이나 한 사람만으로는 동작하지 않습니다.\n"
            "대리 분류기를 학습시키며 잡음을 만들기 때문에, 신원이 하나면 교차 엔트로피 손실이\n"
            "항상 0 이 되어 아무 잡음도 생성되지 않습니다.\n"
            "  --input 에 두 사람 이상을 두거나, --refs 로 참조 신원을 함께 지정하세요.")

    entries, label_of = [], {}
    for name, files in list(protect_groups.items()) + list(reference_groups.items()):
        label_of[name] = len(label_of)
        for path in files:
            entries.append((path, name, name in protect_groups))
    paths = [path for path, _, _ in entries]
    log(f"신원 {num_classes}명 (보호 {len(protect_groups)} / 참조 {len(reference_groups)}), 사진 {len(paths)}장")

    log("얼굴 영역 결정 중...")
    if args.boxes:
        boxes, skipped = load_boxes(args.boxes, paths, args.min_face)
    else:
        boxes, skipped = detect_boxes(paths, device, args.min_face)
    for path, reason in skipped:
        log(f"  건너뜀: {path.name} -- {reason}")
    entries = [e for e in entries if e[0] in boxes]
    if not entries:
        raise SystemExit("얼굴을 찾은 사진이 없습니다.")
    if len({label_of[name] for _, name, _ in entries}) < 2:
        raise SystemExit("얼굴을 찾은 사진의 신원이 1명뿐입니다. 신원이 둘 이상 필요합니다.")

    faces, labels = [], []
    for path, name, _ in entries:
        x, y, side = boxes[path]
        faces.append(load_photo(path)[:, y:y + side, x:x + side].contiguous())
        labels.append(label_of[name])
    sides = [face.shape[-1] for face in faces]
    log(f"얼굴 {len(faces)}개, 한 변 {min(sides)}~{max(sides)}px")

    log("보호 잡음 생성 중...")
    noise = generate(faces, labels, num_classes, args, device, log)

    output_root = Path(args.output); output_root.mkdir(parents=True, exist_ok=True)
    written, residuals = 0, []
    for (path, name, is_protected), delta in zip(entries, noise):
        if not is_protected:
            continue                                   # reference identities are not published
        x, y, side = boxes[path]
        photo = load_photo(path)
        face = photo[:, y:y + side, x:x + side]
        published = (face + delta).clamp(0, 1)
        residuals.append(float((published - face).abs().max()))
        photo[:, y:y + side, x:x + side] = published
        folder = output_root / name; folder.mkdir(parents=True, exist_ok=True)
        save_photo(photo, folder / (path.stem + ".png"))
        written += 1
    if args.save_noise:
        torch.save(dict(noise=noise, boxes={str(p): boxes[p] for p, _, _ in entries},
                        labels=labels, paths=[str(p) for p, _, _ in entries]), args.save_noise)
        log(f"잡음 저장: {args.save_noise}")

    if written == 0:
        raise SystemExit("보호 사진이 하나도 생성되지 않았습니다. --input 의 사진에서 "
                         "얼굴을 찾지 못했는지 위의 건너뜀 목록을 확인하세요.")
    log(f"\n보호 사진 {written}장 저장: {output_root}")
    if residuals:
        log(f"실제 섭동 최대 {max(residuals) * 255:.2f}/255 (상한 {args.epsilon * 255:.2f}/255)")
    log("출력은 PNG 입니다. JPEG 로 다시 저장하면 압축이 잡음을 깎아 보호가 약해집니다.")
    return written


# -------------------------------------------------------------------- self test

def self_test(args, device):
    """End-to-end run on synthetic photos, checking the invariants that matter."""
    import tempfile
    log = print
    log("자체 점검: 합성 사진으로 전 과정을 돌립니다 (학습이 아니라 동작 확인입니다).\n")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, output = root / "in", root / "out"
        generator = torch.Generator().manual_seed(0)
        originals, boxes = {}, {}
        for person in ("person_a", "person_b"):
            (source / person).mkdir(parents=True)
            for n in range(4):
                photo = torch.rand(3, 200, 160, generator=generator) * .6 + .2
                path = source / person / f"{n}.png"
                save_photo(photo, path)
                originals[path] = load_photo(path)
                boxes[str(path)] = [40, 30, 96]
        box_file = root / "boxes.json"; box_file.write_text(json.dumps(boxes))

        args.input, args.refs, args.output, args.boxes = source, None, output, box_file
        args.epochs, args.steps, args.batch_size, args.min_face = 2, 3, 4, 8
        written = run(args, device, log)

        log("\n불변식 검사")
        failures = []
        if written != 8:
            failures.append(f"보호 사진 수가 8이 아니라 {written}")
        for path, original in originals.items():
            produced = output / path.parent.name / (path.stem + ".png")
            if not produced.exists():
                failures.append(f"출력 없음: {produced.name}"); continue
            got = load_photo(produced)
            if got.shape != original.shape:
                failures.append(f"크기 변경: {produced.name}"); continue
            x, y, side = boxes[str(path)]
            outside = (got - original).abs().clone()
            outside[:, y:y + side, x:x + side] = 0
            inside = (got[:, y:y + side, x:x + side]
                      - original[:, y:y + side, x:x + side]).abs().max()
            if float(outside.max()) > 1.5 / 255:
                failures.append(f"얼굴 영역 밖이 변경됨: {produced.name} "
                                f"({float(outside.max()) * 255:.2f}/255)")
            if float(inside) > args.epsilon + 1.5 / 255:
                failures.append(f"섭동이 상한 초과: {produced.name} "
                                f"({float(inside) * 255:.2f}/255)")
        log("  1. 보호 사진이 모두 생성되었다")
        log("  2. 사진 크기가 보존되었다")
        log("  3. 얼굴 영역 밖의 화소가 바뀌지 않았다 (PNG 양자화 1/255 허용)")
        log("  4. 섭동이 eps 상한 안에 있다")
        if failures:
            log("\n실패:"); [log("  - " + f) for f in failures]
            return 1
        log("\n자체 점검 통과. 실제 사진으로 쓸 수 있습니다.")
        return 0


# ------------------------------------------------------------------------- main

def build_parser():
    parser = argparse.ArgumentParser(
        description="LPID -- 얼굴 사진을 무단 학습으로부터 보호한 사본을 생성한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="신원이 둘 이상 필요합니다 (--input 에 두 사람 이상, 또는 --refs 사용).")
    parser.add_argument("--input", help="보호할 사진 폴더. 하위 폴더 하나가 한 사람.")
    parser.add_argument("--refs", default=None,
                        help="참조 신원 폴더. 최적화에만 쓰이고 출력되지 않는다.")
    parser.add_argument("--output", help="보호 사진을 저장할 폴더.")
    parser.add_argument("--boxes", default=None,
                        help='얼굴 영역 JSON {"경로": [x, y, 한변]}. 지정하면 검출기를 쓰지 않는다.')
    parser.add_argument("--save-noise", default=None, help="생성한 잡음을 저장할 .pt 경로.")
    parser.add_argument("--epsilon", type=float, default=8 / 255, help="섭동 크기 상한 (기본 8/255).")
    parser.add_argument("--step-size", type=float, default=0.8 / 255, help="PGD 단계 크기 (기본 0.8/255).")
    parser.add_argument("--steps", type=int, default=20, help="갱신당 PGD 반복 (기본 20).")
    parser.add_argument("--epochs", type=int, default=60, help="최대 반복 (기본 60).")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기 (기본 32).")
    parser.add_argument("--target-size", type=int, default=224,
                        help="공격자가 확대할 입력 크기 (기본 224).")
    parser.add_argument("--lr", type=float, default=0.025, help="대리 모델 학습률 (기본 0.025).")
    parser.add_argument("--stop-acc", type=float, default=0.99, help="조기 종료 기준 (기본 0.99).")
    parser.add_argument("--arch", default="resnet18", help="대리 모델 (기본 resnet18).")
    parser.add_argument("--min-face", type=int, default=32, help="이보다 작은 얼굴은 건너뛴다 (기본 32px).")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드 (기본 42).")
    parser.add_argument("--device", default=None, help="cuda 또는 cpu. 기본은 자동 선택.")
    parser.add_argument("--self-test", action="store_true",
                        help="합성 사진으로 전 과정을 점검하고 종료한다.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")
    if args.self_test:
        return self_test(args, device)
    missing = [name for name in ("input", "output") if not getattr(args, name)]
    if missing:
        raise SystemExit("필수 인자가 없습니다: " + ", ".join("--" + m for m in missing))
    run(args, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
