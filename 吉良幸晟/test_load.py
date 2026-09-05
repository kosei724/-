from dataloader import load_data

df = load_data("all", panda=True, includeNan=True, fillNan=True)

print("形状:", df.shape)
print()
print("試合ごとの行数:")
print(df["game"].value_counts())
print()
print("試合ごとのタックル数:")
print(df[df["is_tackle"] == 1]["game"].value_counts())
print()
print("タックルの割合:", f'{df["is_tackle"].mean()*100:.2f}%')