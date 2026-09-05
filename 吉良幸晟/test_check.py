from dataloader import load_data

df = load_data("all", panda=True, includeNan=True, fillNan=True)

# ricohのタックル行を確認
ricoh = df[df["game"] == "ricoh"].reset_index(drop=True)
tackle_idx = ricoh[ricoh["is_tackle"] == 1].index.tolist()

print("タックル行のインデックス:")
print(tackle_idx)
print()

# 連続した区間に分ける
groups = []
start = tackle_idx[0]
prev = tackle_idx[0]
for i in tackle_idx[1:]:
    if i - prev > 1:
        groups.append((start, prev, prev - start + 1))
        start = i
    prev = i
groups.append((start, prev, prev - start + 1))

print("タックル区間（開始行, 終了行, 長さ）:")
for g in groups:
    print(g)