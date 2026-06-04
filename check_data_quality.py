import pandas as pd

# Load data
df = pd.read_csv('data/member_DU2.csv')

labels = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 
          'anticipation', 'love', 'worried', 'disappointed', 'proud', 
          'embarrassed', 'jealous', 'calm', 'excited']

print("=" * 70)
print("DATA QUALITY CHECK - member_DU2.csv")
print("=" * 70)
print(f"\nTotal samples: {len(df)}")

# Check label distribution
print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)
for label in labels:
    count = df[label].sum()
    pct = (count / len(df)) * 100
    print(f"{label:15s}: {count:5d} samples ({pct:5.1f}%)")

# Check for problematic labels (too few samples)
print("\n" + "=" * 70)
print("PROBLEMATIC LABELS (< 1% of data)")
print("=" * 70)
for label in labels:
    count = df[label].sum()
    pct = (count / len(df)) * 100
    if pct < 1.0:
        print(f"⚠️  {label:15s}: {count:5d} samples ({pct:5.1f}%) - TOO FEW!")

# Check text quality
print("\n" + "=" * 70)
print("TEXT QUALITY")
print("=" * 70)
print(f"Empty texts: {df['text'].isna().sum()}")
print(f"Very short texts (< 5 chars): {(df['text'].str.len() < 5).sum()}")
print(f"Average text length: {df['text'].str.len().mean():.1f} chars")

# Check for duplicates
print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)
duplicates = df['text'].duplicated().sum()
print(f"Duplicate texts: {duplicates} ({duplicates/len(df)*100:.1f}%)")

# Check label combinations
df['total_labels'] = df[labels].sum(axis=1)
print("\n" + "=" * 70)
print("LABEL COMBINATIONS")
print("=" * 70)
print(f"Samples with 0 labels: {(df['total_labels'] == 0).sum()}")
print(f"Samples with 1 label: {(df['total_labels'] == 1).sum()}")
print(f"Samples with 2 labels: {(df['total_labels'] == 2).sum()}")
print(f"Samples with 3+ labels: {(df['total_labels'] >= 3).sum()}")
print(f"Average labels per sample: {df['total_labels'].mean():.2f}")
print(f"Max labels per sample: {df['total_labels'].max()}")

# Sample some data
print("\n" + "=" * 70)
print("SAMPLE DATA")
print("=" * 70)
print("\nFirst 10 texts:")
for i, row in df.head(10).iterrows():
    active_labels = [label for label in labels if row[label] == 1]
    print(f"{i+1}. {row['text'][:60]}... -> {active_labels}")
