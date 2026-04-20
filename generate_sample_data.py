"""
Sample Data Generation Script for Multi-label Emotion Classification

This script generates sample comment data with multi-label emotion annotations
for demonstrating the emotion classification system.

Generates:
- At least 100 sample comments
- Mix of English and Vietnamese text
- Realistic multi-label emotion combinations
- Saves to data/sample_comments.csv
"""

import pandas as pd
import random
from config import Config


def generate_sample_data(num_samples=100):
    """
    Generate sample comments with multi-label emotion annotations.
    
    Args:
        num_samples (int): Number of samples to generate (default 100)
    
    Returns:
        pd.DataFrame: DataFrame with 'text' column and 16 emotion label columns
    """
    # Set random seed for reproducibility
    random.seed(Config.RANDOM_SEED)
    
    # Sample comments with realistic emotion combinations
    # Format: (text, [emotion_indices])
    # Emotion indices correspond to Config.EMOTION_LABELS order
    
    sample_templates = [
        # Joy-based combinations
        ("This product is absolutely amazing! I love it so much!", [0, 1, 8, 15]),  # joy, trust, love, excited
        ("I'm so happy with my purchase! Exceeded expectations!", [0, 1, 8, 15]),  # joy, trust, love, excited
        ("Best decision ever! I'm thrilled!", [0, 8, 11, 15]),  # joy, love, proud, excited
        ("Tôi rất vui và hạnh phúc với sản phẩm này!", [0, 8, 15]),  # joy, love, excited
        ("Tuyệt vời! Tôi yêu nó!", [0, 1, 8]),  # joy, trust, love
        
        # Sadness/Disappointment combinations
        ("I'm really disappointed with the quality. Very sad.", [4, 5, 10]),  # sadness, disgust, disappointed
        ("This made me so sad. Not what I expected at all.", [4, 10]),  # sadness, disappointed
        ("Feeling let down and disappointed by this purchase.", [4, 6, 10]),  # sadness, anger, disappointed
        ("Tôi rất thất vọng và buồn về sản phẩm này.", [4, 10]),  # sadness, disappointed
        
        # Anger combinations
        ("This is terrible! I'm so angry and frustrated!", [5, 6, 10]),  # disgust, anger, disappointed
        ("Absolutely furious with this poor service!", [5, 6]),  # disgust, anger
        ("I'm disgusted and angry. Complete waste of money!", [5, 6, 10]),  # disgust, anger, disappointed
        ("Tôi rất tức giận! Dịch vụ tệ!", [6, 10]),  # anger, disappointed
        
        # Fear/Worry combinations
        ("I'm worried this won't work properly. Feeling anxious.", [2, 9]),  # fear, worried
        ("Concerned and fearful about the safety of this product.", [2, 9]),  # fear, worried
        ("I'm scared and worried it might break easily.", [2, 9]),  # fear, worried
        ("Tôi lo lắng về chất lượng sản phẩm.", [2, 9]),  # fear, worried
        
        # Surprise combinations
        ("Wow! I'm surprised by how good this is!", [0, 3, 15]),  # joy, surprise, excited
        ("Didn't expect this! Pleasantly surprised!", [0, 3, 1]),  # joy, surprise, trust
        ("I'm shocked at how amazing this turned out!", [0, 3, 8]),  # joy, surprise, love
        ("Thật ngạc nhiên! Tốt hơn tôi nghĩ!", [3, 0]),  # surprise, joy
        
        # Anticipation combinations
        ("Can't wait to use this! So excited and hopeful!", [7, 15]),  # anticipation, excited
        ("Looking forward to seeing the results. Feeling optimistic!", [7, 1]),  # anticipation, trust
        ("I'm eager and excited to try this out!", [7, 15]),  # anticipation, excited
        ("Tôi rất mong chờ! Hào hứng quá!", [7, 15]),  # anticipation, excited
        
        # Calm/Trust combinations
        ("Everything works as expected. I'm satisfied and calm.", [1, 14]),  # trust, calm
        ("Reliable product. Feeling peaceful and content.", [1, 14]),  # trust, calm
        ("No issues so far. Calm and trusting the quality.", [1, 14]),  # trust, calm
        ("Sản phẩm ổn định. Tôi tin tưởng.", [1, 14]),  # trust, calm
        
        # Proud combinations
        ("I'm so proud of this purchase! Great choice!", [0, 11, 1]),  # joy, proud, trust
        ("Feeling proud and satisfied with my decision.", [11, 0, 14]),  # proud, joy, calm
        ("This makes me proud to own it!", [11, 0, 8]),  # proud, joy, love
        
        # Embarrassed combinations
        ("I'm a bit embarrassed I didn't buy this sooner.", [12, 10]),  # embarrassed, disappointed
        ("Feeling embarrassed about my initial doubts.", [12]),  # embarrassed
        ("Slightly embarrassed to admit I was wrong about this.", [12, 3]),  # embarrassed, surprise
        
        # Jealous combinations
        ("I'm jealous of people who discovered this earlier!", [13, 9]),  # jealous, worried
        ("Feeling a bit jealous seeing others enjoy this.", [13, 4]),  # jealous, sadness
        ("Wish I had found this sooner. Slightly jealous.", [13, 10]),  # jealous, disappointed
        
        # Mixed complex emotions
        ("I'm excited but also worried about the outcome.", [15, 9, 7]),  # excited, worried, anticipation
        ("Happy yet surprised by the unexpected features.", [0, 3, 8]),  # joy, surprise, love
        ("Disappointed but hopeful it will improve.", [10, 7, 9]),  # disappointed, anticipation, worried
        ("Angry but also sad about this situation.", [6, 4, 10]),  # anger, sadness, disappointed
        ("Tôi vui nhưng cũng lo lắng một chút.", [0, 9]),  # joy, worried
        ("Hạnh phúc nhưng ngạc nhiên về giá cả.", [0, 3, 8]),  # joy, surprise, love
        
        # Neutral/Single emotion
        ("The product is okay. Nothing special.", [14]),  # calm
        ("It works fine. No complaints.", [1, 14]),  # trust, calm
        ("Standard quality. As described.", [1]),  # trust
        ("Sản phẩm bình thường.", [14]),  # calm
        
        # More English variations
        ("Absolutely love this! Best purchase ever!", [0, 8, 11, 15]),
        ("So disappointed and frustrated. Waste of money.", [10, 6, 4]),
        ("I'm thrilled and can't contain my excitement!", [0, 15, 8]),
        ("Feeling anxious and fearful about using this.", [2, 9]),
        ("Pleasantly surprised! Exceeded all expectations!", [3, 0, 1]),
        ("Calm and satisfied with the performance.", [14, 1]),
        ("I'm proud to own this quality product!", [11, 0, 1]),
        ("Disgusted by the poor quality and service.", [5, 6, 10]),
        ("Worried this might not last long.", [9, 2]),
        ("Excited to see what happens next!", [15, 7]),
        
        # More Vietnamese variations
        ("Tôi rất hài lòng và tin tưởng sản phẩm này!", [0, 1, 14]),
        ("Thất vọng và tức giận với chất lượng.", [10, 6, 4]),
        ("Ngạc nhiên và vui mừng với kết quả!", [3, 0, 15]),
        ("Lo lắng về độ bền của sản phẩm.", [9, 2]),
        ("Tự hào về quyết định mua hàng này!", [11, 0]),
        ("Bình tĩnh và hài lòng với hiệu suất.", [14, 1]),
        ("Hào hứng chờ đợi sử dụng!", [15, 7]),
        ("Buồn và thất vọng về dịch vụ.", [4, 10]),
        ("Yêu thích sản phẩm này! Tuyệt vời!", [8, 0, 15]),
        ("Ghê tởm chất lượng kém này.", [5, 6]),
        
        # Additional diverse examples
        ("I trust this brand completely. Very reliable.", [1, 14]),
        ("Fearful of potential issues down the line.", [2, 9]),
        ("Surprised by the fast delivery! Happy customer!", [3, 0, 1]),
        ("Anticipating great results from this purchase.", [7, 1, 15]),
        ("Feeling disgusted and let down by the experience.", [5, 10, 4]),
        ("Calm and peaceful using this product.", [14]),
        ("Excited and joyful about this amazing find!", [15, 0, 8]),
        ("Embarrassed I doubted the quality initially.", [12, 3]),
        ("Jealous of those who got the better deal.", [13, 10]),
        ("Proud and happy with my smart choice!", [11, 0, 1]),
        ("Worried and anxious about the warranty.", [9, 2]),
        ("Disappointed yet hopeful for improvements.", [10, 7]),
        ("Angry and frustrated with customer service.", [6, 5, 10]),
        ("Loving every moment with this product!", [8, 0, 15]),
        ("Trusting the process and staying calm.", [1, 14]),
        
        # More Vietnamese examples
        ("Tin tưởng hoàn toàn vào thương hiệu này.", [1, 14]),
        ("Sợ hãi về các vấn đề tiềm ẩn.", [2, 9]),
        ("Ngạc nhiên về giao hàng nhanh! Vui lắm!", [3, 0, 1]),
        ("Mong đợi kết quả tuyệt vời từ sản phẩm.", [7, 1, 15]),
        ("Cảm thấy ghê tởm và thất vọng.", [5, 10, 4]),
        ("Bình tĩnh và yên bình khi sử dụng.", [14]),
        ("Hào hứng và vui mừng với phát hiện tuyệt vời!", [15, 0, 8]),
        ("Xấu hổ vì đã nghi ngờ chất lượng.", [12, 3]),
        ("Ghen tị với những người có giá tốt hơn.", [13, 10]),
        ("Tự hào và hạnh phúc với lựa chọn thông minh!", [11, 0, 1]),
        ("Lo lắng và căng thẳng về bảo hành.", [9, 2]),
        ("Thất vọng nhưng vẫn hy vọng cải thiện.", [10, 7]),
        ("Tức giận và bực bội với dịch vụ khách hàng.", [6, 5, 10]),
        ("Yêu thích từng khoảnh khắc với sản phẩm!", [8, 0, 15]),
        ("Tin tưởng vào quá trình và giữ bình tĩnh.", [1, 14]),
    ]
    
    # Generate data
    data = []
    
    # Use all templates first
    for text, emotion_indices in sample_templates:
        # Create binary label vector
        labels = [0] * Config.NUM_LABELS
        for idx in emotion_indices:
            labels[idx] = 1
        
        data.append([text] + labels)
    
    # If we need more samples, generate variations
    while len(data) < num_samples:
        # Randomly select a template and add slight variation
        text, emotion_indices = random.choice(sample_templates)
        
        # Create binary label vector
        labels = [0] * Config.NUM_LABELS
        for idx in emotion_indices:
            labels[idx] = 1
        
        data.append([text] + labels)
    
    # Create DataFrame
    columns = ['text'] + Config.EMOTION_LABELS
    df = pd.DataFrame(data[:num_samples], columns=columns)
    
    return df


def main():
    """
    Main function to generate and save sample data.
    """
    print("Generating sample emotion classification data...")
    print(f"Target: {100} samples minimum")
    print(f"Languages: English and Vietnamese")
    print(f"Emotions: {len(Config.EMOTION_LABELS)} labels")
    print()
    
    # Generate data
    df = generate_sample_data(num_samples=100)
    
    # Display statistics
    print(f"Generated {len(df)} samples")
    print(f"\nEmotion distribution:")
    for emotion in Config.EMOTION_LABELS:
        count = df[emotion].sum()
        percentage = (count / len(df)) * 100
        print(f"  {emotion:15s}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Calculate average emotions per comment
    avg_emotions = df[Config.EMOTION_LABELS].sum(axis=1).mean()
    print(f"\nAverage emotions per comment: {avg_emotions:.2f}")
    
    # Verify all comments have at least one emotion
    min_emotions = df[Config.EMOTION_LABELS].sum(axis=1).min()
    print(f"Minimum emotions per comment: {min_emotions}")
    
    # Save to CSV
    output_path = Config.DATA_DIR + "sample_comments.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSample data saved to: {output_path}")
    print("\nSample rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
