import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----- File Paths -----
csv_path = "/home/nandkumar/Desktop/Research Projet/emotion_log.csv"
charts_dir = "/home/nandkumar/Desktop/Research Projet/charts"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ----- Ensure directories exist -----
os.makedirs(charts_dir, exist_ok=True)

# ----- Load Data -----
if not os.path.exists(csv_path):
    print(f"❌ CSV file not found: {csv_path}")
    exit()

df = pd.read_csv(csv_path)

if 'Emotion' not in df.columns or 'Time' not in df.columns:
    print("❌ Required columns 'Emotion' or 'Time' not found in CSV.")
    exit()

# ----- Normalize Time Format -----
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
df = df.dropna(subset=['Time'])

# ----- Emotion Distribution Pie -----
emotion_counts = df['Emotion'].value_counts()
plt.figure(figsize=(8, 8))
emotion_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title("Detected Emotion Distribution")
plt.ylabel("")
plt.tight_layout()
pie_path = f"{charts_dir}/emotion_distribution_pie_{timestamp}.png"
plt.savefig(pie_path)
print(f"✅ Saved: {pie_path}")
plt.close()

# ----- Emotion Distribution Bar -----
plt.figure(figsize=(10, 6))
emotion_counts.plot(kind='bar', color=plt.cm.Paired.colors)
plt.title("Emotion Frequency")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
bar_path = f"{charts_dir}/emotion_distribution_bar_{timestamp}.png"
plt.savefig(bar_path)
print(f"✅ Saved: {bar_path}")
plt.close()

# ----- Updated Engagement Classification -----
engaged_emotions = ["Happy", "Surprised", "Neutral"]
df['Engagement'] = df['Emotion'].apply(lambda x: 'Engaged' if x in engaged_emotions else 'Disengaged')
engagement_counts = df['Engagement'].value_counts()

# ----- Engagement Pie -----
plt.figure(figsize=(6, 6))
engagement_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
plt.title("Engaged vs Disengaged")
plt.ylabel("")
plt.tight_layout()
engage_pie_path = f"{charts_dir}/engagement_pie_{timestamp}.png"
plt.savefig(engage_pie_path)
print(f"✅ Saved: {engage_pie_path}")
plt.close()

# ----- Emotion Over Time Line Chart -----
time_emotion_counts = df.groupby([df['Time'].dt.floor('min'), 'Emotion']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
time_emotion_counts.plot(marker='o')
plt.title("Emotion Trend Over Time")
plt.xlabel("Time (minute)")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
line_chart_path = f"{charts_dir}/emotion_trend_{timestamp}.png"
plt.savefig(line_chart_path)
print(f"✅ Saved: {line_chart_path}")
plt.close()

# ----- Engagement Trend Over Time (Stacked Area Chart) -----
engagement_over_time = df.groupby([df['Time'].dt.floor('min'), 'Engagement']).size().unstack(fill_value=0)
engagement_over_time.plot(kind='area', stacked=True, figsize=(12, 6), colormap='Set2')
plt.title("Engagement Trend Over Time")
plt.xlabel("Time (minute)")
plt.ylabel("Count")
plt.tight_layout()
engage_trend_path = f"{charts_dir}/engagement_trend_{timestamp}.png"
plt.savefig(engage_trend_path)
print(f"✅ Saved: {engage_trend_path}")
plt.close()


# ----- Summary Statistics Export -----
summary_stats = emotion_counts.reset_index()
summary_stats.columns = ['Emotion', 'Count']
summary_csv = f"{charts_dir}/emotion_summary_{timestamp}.csv"
summary_stats.to_csv(summary_csv, index=False)
print(f"✅ Saved summary CSV: {summary_csv}")

# ----- Confusion Matrix  -----
if 'Actual' in df.columns and 'Predicted' in df.columns:
    try:
        cm_labels = sorted(df['Actual'].unique())
        cm = confusion_matrix(df['Actual'], df['Predicted'], labels=cm_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = f"{charts_dir}/confusion_matrix_{timestamp}.png"
        plt.savefig(cm_path)
        print(f"✅ Saved: {cm_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️ Confusion matrix could not be plotted: {e}")