"""Numbered labelling sheets: 24 crops per sheet, upper 70% of each crop (the riders) enlarged."""
import sys, json, cv2
D = sys.argv[1]
meta = json.load(open(f"{D}/meta.json"))
meta.sort(key=lambda m: (m["split"], m["t"]))
for i, m in enumerate(meta): m["idx"] = i
json.dump(meta, open(f"{D}/meta.json", "w"), indent=0)
TW, TH = 240, 300
for s in range(0, len(meta), 24):
    tiles = []
    for m in meta[s:s + 24]:
        im = cv2.imread(f"{D}/crops/{m['file']}")
        im = im[: int(im.shape[0] * 0.75)]
        h, w = im.shape[:2]; k = min(TW / w, TH / h)
        im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_CUBIC)
        tile = cv2.copyMakeBorder(im, 0, TH - im.shape[0], 0, TW - im.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.rectangle(tile, (0, 0), (52, 26), (0, 0, 0), -1)
        cv2.putText(tile, str(m["idx"]), (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        tiles.append(cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % 6: tiles.append(tiles[0] * 0)
    rows = [cv2.hconcat(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(f"{D}/sheet_{s // 24:02d}.jpg", cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
print(len(meta), "crops,", (len(meta) + 23) // 24, "sheets")
