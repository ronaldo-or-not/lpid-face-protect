# Transform-Robust Face Photo Protection (LPID)

Version 1.0 / 2026-10-31

Author: Byunghoon Oh (Dept. of Convergence Security, Chung-Ang University)


## 1. Overview

Given a user's face photos, the program outputs protected photos that look the
same to the eye but, even if scraped without consent and used to train a
face-recognition model, prevent that model from recognizing the user.

Existing unlearnable-example methods generate the protective noise over the
full-resolution image. The standard face-recognition pipeline, however, crops
the face region and upscales it to the model's input size; the interpolation in
that transform wipes out the high-frequency components of the noise, so the
protection is lost.

This program (1) confines the protective noise to the face region the attacker
will crop, and (2) implements that crop-and-upscale transform as a
differentiable operation, optimizing the noise *through* the transform. Because
the gradient of the post-transform loss backpropagates into the noise, the
optimization does not spend its perturbation budget on components that the
transform would remove, and instead concentrates energy in the low-frequency
band that survives the transform.


## 2. Files

    protect.py           main program (406 lines)
    requirements.txt     dependency list
    source_listing.txt   full source listing for submission (with line numbers)
    README.md            this document


## 3. Environment

    OS         Linux (verified on Ubuntu 20.04+)
    Language   Python 3.10+
    Hardware   NVIDIA GPU (CUDA) recommended. CPU-only also works but is slow.

See requirements.txt for dependencies.

    pip install -r requirements.txt


## 4. Usage

### 4.1 Self-test (recommended once, on first use)

    python3 protect.py --self-test

Runs the whole pipeline on synthetic photos and checks the following four
properties, then prints the results.

    1. All protected photos were generated
    2. Photo resolution was preserved
    3. Pixels outside the face region were not changed
    4. The perturbation stays within the specified bound

### 4.2 Photo folder layout

Create one subfolder per person and place that person's photos inside.

    my_photos/
      hong/   001.jpg  002.jpg  ...
      kim/    001.jpg  002.jpg  ...

### 4.3 Run

To protect two or more people:

    python3 protect.py --input ./my_photos --output ./protected

To protect a single person (provide a reference-identity folder as well):

    python3 protect.py --input ./my_photos --refs ./reference_ids --output ./protected

See all arguments with:

    python3 protect.py --help


## 5. Key arguments

    --epsilon       perturbation bound (default 8/255 = 0.0313)
    --step-size     PGD step size (default 0.8/255)
    --steps         PGD iterations per update (default 20)
    --epochs        maximum number of epochs (default 60)
    --batch-size    batch size (default 32)
    --target-size   attacker's upscale input size (default 224)
    --lr            surrogate-model learning rate (default 0.025)
    --stop-acc      early-stopping criterion (default 0.99)
    --arch          surrogate model (default resnet18)
    --boxes         specify face regions directly as JSON (no detector)
    --min-face      skip faces smaller than this (default 32px)
    --save-noise    save the generated noise to a file
    --seed          random seed (default 42)
    --device        cuda or cpu


## 6. Notes on use

### 6.1 At least two identities are required

The program generates the protective noise while training a surrogate
classifier. The input must therefore be several photos per person, not a single
photo, and there must be at least two identities. With only one identity the
cross-entropy loss is always 0, so no noise is generated; in that case the
program explains why and stops.

To protect a single person, provide a reference-identity folder with --refs. The
reference identities' photos are used only to generate the noise and are not
output.

### 6.2 Outputs are lossless PNG

Re-saving with a lossy codec such as JPEG trims the protective noise during
compression and weakens the protection. Keep protected photos as PNG.

### 6.3 Reproducibility

The random seed and cuDNN deterministic settings are fixed, so the same input
and the same seed reproduce the same protective noise.


## 7. Algorithm summary

Inner optimization (protective noise)

    Sign-based projected gradient descent in the direction that decreases the
    surrogate model's classification loss, 20 iterations per update.
    Perturbation bound 8/255, step size 0.8/255. At each step the protected
    photo's pixel values are projected back into the valid range.

Outer optimization (surrogate model)

    A ResNet-18 surrogate model is updated with stochastic gradient descent.
    Learning rate 0.025, momentum 0.9, weight decay 5e-4, cosine annealing.
    Up to 60 epochs, with early stopping once training accuracy reaches 0.99.

Extraction-transform model

    Face-region cropping and bilinear upscaling to the target input size are
    implemented as differentiable operations. The protective noise is stored per
    photo at the face-region resolution, so input photos of different sizes are
    handled.
