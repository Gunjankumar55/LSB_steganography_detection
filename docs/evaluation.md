# Evaluation notes

These notes record what the published models do when the shipped pipeline is
run on clean and payload-bearing images. They exist because the repository
contains no training code, no dataset, and no held-out metrics, so any claim
about detection quality has to come from a measurement.

**This is a spot check, not a benchmark.** It covers 28 images in total, uses
one naive embedding routine, and says nothing about the data the models were
trained on.

Measured on 2026-11-19 with the artifacts from release `v1.0.0`.

## Method

1. The image is converted to RGB, resized to 256x256 with Lanczos, and turned
   into the 196,614-value feature vector described in
   [`steganalysis/features.py`](../steganalysis/features.py).
2. The vector is scaled with `final_scaler.pkl` and passed to both networks.
3. The binary classifier decides clean or stego. The payload classifier only
   runs when the binary classifier answers "stego".
4. Payloads are embedded at the image's native resolution into the two least
   significant bits of the flattened RGB plane, in sequence, with no length
   header, and the result is PNG-encoded. This is a naive LSB embedding and it
   is unlikely to match the routine used to build the training corpus.
5. The clean control is the same image before embedding.

A stego row counts as a binary success when the model answers "stego"; a clean
row counts as a success when it answers "clean". Payload typing is scored only
on rows where a payload was actually embedded.

## Results

### Synthetic images

Reproduce with:

```sh
python eval/spot_check.py
```

Three generated images, each run clean and with an HTML, JavaScript, and
PowerShell payload.

| Image | Clean images called stego | Embedded images caught | Payload type correct |
| --- | --- | --- | --- |
| gradient | 1/1 | 3/3 | 1/3 |
| flat colour | 1/1 | 3/3 | 1/3 |
| smooth pattern | 1/1 | 3/3 | 0/3 |
| **total** | **3/3** | **9/9** | **2/9** |

Binary result: 9/12 correct. All three errors are clean images labelled stego,
at 0.61 to 0.90 confidence.

### Photographic images

Reproduce with a directory of your own images:

```sh
python eval/spot_check.py --images path/to/images --limit 4
```

Four photographs (Windows wallpaper set, `ThemeA/img20` to `img23`), each run
clean and with the three payloads.

| Image | Clean images called stego | Embedded images caught | Payload type correct |
| --- | --- | --- | --- |
| img20 | 1/1 | 3/3 | 0/3 |
| img21 | 1/1 | 0/3 | 0/3 |
| img22 | 1/1 | 3/3 | 1/3 |
| img23 | 1/1 | 3/3 | 0/3 |
| **total** | **4/4** | **9/12** | **1/12** |

Binary result: 9/16 correct.

## What the numbers show

- The binary classifier answers "stego" for every clean image in both sets. The
  only clean image that came back correct during the wider probe was a flat
  graphic logo, scored 0.999 clean.
- Embedded images are usually flagged as stego, at 0.97 to 1.00 confidence, so
  the model does respond to the embedding signal. The decision boundary sits
  too far toward "stego" for these inputs.
- Payload typing is not reliable. On stego rows it predicted `clean` in 9 of 12
  photographic cases and `js` for every synthetic case where the payload was
  HTML or PowerShell.
- Changing the resize filter does not explain the bias. Lanczos, bilinear,
  bicubic, nearest, box, and hamming all produced clean-image stego
  probabilities within 0.05 of each other.

## Caveats

- 28 image runs, drawn from two small sets. Confidence intervals on three or
  four clean images are enormous.
- The embedding routine is a guess. A different bit order, payload length, or
  LSB count could move the results substantially in either direction.
- The photographs were JPEG-compressed at some point in their life. JPEG noise
  already fills the least significant bits, which is a plausible reason a
  clean photo looks like a carrier.
- The verification machine ran scikit-learn 1.4.2 against artifacts serialised
  with 1.7.2, which emits `InconsistentVersionWarning`. Re-running the original
  `app.py` and the refactored pipeline on the same RGB image produced the same
  probabilities to within 1e-4, so the version gap did not visibly change the
  output here.
- No training data, training script, or held-out evaluation is present in the
  repository, so the models cannot be retrained or properly evaluated from it.

## Conclusion

The project is a working end-to-end demo: upload, feature extraction, two
PyTorch models, JSON response, and a UI. It is not a validated steganalysis
detector. Treat the probabilities as illustrative.
