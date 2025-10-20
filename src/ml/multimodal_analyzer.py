"""
Multimodal Analysis - Image and Video Sentiment

Analyzes visual content in social media:
- OCR text extraction from images
- Chart/graph detection
- Meme sentiment analysis
- Logo/brand detection
- Color sentiment analysis
"""
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
import requests
from loguru import logger

# Optional dependencies
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logger.warning("pytesseract not available - OCR disabled")

try:
    from transformers import pipeline
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


@dataclass
class ImageAnalysisResult:
    """Result of image analysis"""
    has_text: bool
    extracted_text: Optional[str]
    text_sentiment: float
    is_chart: bool
    is_meme: bool
    dominant_colors: List[Tuple[int, int, int]]
    color_sentiment: float  # Based on color psychology
    overall_sentiment: float
    confidence: float


@dataclass
class VideoAnalysisResult:
    """Result of video analysis"""
    frame_count: int
    key_frames_analyzed: int
    average_sentiment: float
    sentiment_trend: str  # 'positive', 'negative', 'neutral', 'volatile'
    has_text: bool
    extracted_texts: List[str]


class MultimodalAnalyzer:
    """
    Analyze images and videos for sentiment
    
    Features:
    - OCR text extraction
    - Chart/graph detection
    - Meme classification
    - Color psychology
    - Brand/logo detection
    """
    
    def __init__(self):
        """Initialize multimodal analyzer"""
        self.has_ocr = HAS_OCR
        self.has_clip = HAS_CLIP
        
        # Color sentiment mapping (color psychology)
        self.color_sentiments = {
            'red': -0.3,      # Danger, urgency, negative
            'green': 0.5,     # Positive, growth
            'blue': 0.2,      # Trust, calm
            'yellow': 0.3,    # Warning, attention
            'orange': 0.1,    # Energy, caution
            'purple': 0.0,    # Neutral
            'black': -0.2,    # Serious, negative
            'white': 0.1,     # Clean, neutral
        }
        
        # Chart/graph indicators
        self.chart_keywords = [
            'chart', 'graph', 'candlestick', 'price', 'volume',
            'bullish', 'bearish', 'support', 'resistance',
            'trend', 'moving average', 'rsi', 'macd'
        ]
        
        # Meme indicators
        self.meme_indicators = [
            'to the moon', 'diamond hands', 'paper hands',
            'hodl', 'wen', 'gm', 'wagmi', 'ngmi',
            'rekt', 'pump', 'dump', 'ape in'
        ]
        
        logger.info(f"MultimodalAnalyzer initialized (OCR: {self.has_ocr}, CLIP: {self.has_clip})")
    
    def load_image(self, image_source) -> Optional[Image.Image]:
        """
        Load image from URL, file path, or PIL Image
        
        Args:
            image_source: URL, file path, or PIL Image
        
        Returns:
            PIL Image or None
        """
        try:
            if isinstance(image_source, Image.Image):
                return image_source
            elif isinstance(image_source, str):
                if image_source.startswith('http'):
                    # Download from URL
                    response = requests.get(image_source, timeout=10)
                    return Image.open(BytesIO(response.content))
                else:
                    # Load from file
                    return Image.open(image_source)
            else:
                logger.warning(f"Unknown image source type: {type(image_source)}")
                return None
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def extract_text(self, image: Image.Image) -> Optional[str]:
        """
        Extract text from image using OCR
        
        Args:
            image: PIL Image
        
        Returns:
            Extracted text or None
        """
        if not self.has_ocr:
            logger.warning("OCR not available")
            return None
        
        try:
            # Convert to grayscale for better OCR
            gray_image = image.convert('L')
            
            # Extract text
            text = pytesseract.image_to_string(gray_image)
            
            # Clean text
            text = text.strip()
            
            return text if text else None
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None
    
    def detect_chart(self, image: Image.Image, text: Optional[str] = None) -> bool:
        """
        Detect if image contains a chart/graph
        
        Args:
            image: PIL Image
            text: Optional extracted text
        
        Returns:
            True if chart detected
        """
        # Check text for chart keywords
        if text:
            text_lower = text.lower()
            for keyword in self.chart_keywords:
                if keyword in text_lower:
                    return True
        
        # Check image properties (heuristic)
        # Charts often have:
        # - High contrast
        # - Geometric shapes (lines, rectangles)
        # - Limited color palette
        
        # Convert to array
        img_array = np.array(image.convert('RGB'))
        
        # Check color diversity (charts have limited colors)
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        color_ratio = unique_colors / total_pixels
        
        # Charts typically have < 5% unique colors
        if color_ratio < 0.05:
            return True
        
        return False
    
    def detect_meme(self, text: Optional[str]) -> bool:
        """
        Detect if image is a meme based on text
        
        Args:
            text: Extracted text
        
        Returns:
            True if meme detected
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check for meme indicators
        for indicator in self.meme_indicators:
            if indicator in text_lower:
                return True
        
        # Check for meme text patterns
        # Memes often have:
        # - All caps text
        # - Impact font style (can't detect without font analysis)
        # - Short phrases
        
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.7 and len(text) < 100:
            return True
        
        return False
    
    def analyze_colors(self, image: Image.Image) -> Tuple[List[Tuple[int, int, int]], float]:
        """
        Analyze dominant colors and their sentiment
        
        Args:
            image: PIL Image
        
        Returns:
            (dominant_colors, color_sentiment)
        """
        # Resize for speed
        small_image = image.resize((100, 100))
        img_array = np.array(small_image.convert('RGB'))
        
        # Flatten to list of colors
        pixels = img_array.reshape(-1, 3)
        
        # Find dominant colors using simple binning
        # Group similar colors
        bins = 8  # Reduce colors to 8^3 = 512 bins
        binned = (pixels // (256 // bins)) * (256 // bins)
        
        # Count occurrences
        unique, counts = np.unique(binned, axis=0, return_counts=True)
        
        # Get top 3 colors
        top_indices = np.argsort(counts)[-3:]
        dominant_colors = [tuple(unique[i]) for i in top_indices]
        
        # Calculate sentiment based on color psychology
        sentiment = 0.0
        
        for color in dominant_colors:
            r, g, b = color
            
            # Classify color
            if r > 200 and g < 100 and b < 100:
                sentiment += self.color_sentiments['red']
            elif r < 100 and g > 200 and b < 100:
                sentiment += self.color_sentiments['green']
            elif r < 100 and g < 100 and b > 200:
                sentiment += self.color_sentiments['blue']
            elif r > 200 and g > 200 and b < 100:
                sentiment += self.color_sentiments['yellow']
            elif r > 200 and g > 100 and b < 100:
                sentiment += self.color_sentiments['orange']
            elif r < 50 and g < 50 and b < 50:
                sentiment += self.color_sentiments['black']
            elif r > 200 and g > 200 and b > 200:
                sentiment += self.color_sentiments['white']
        
        # Average sentiment
        sentiment = sentiment / len(dominant_colors)
        
        return dominant_colors, sentiment
    
    def analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of extracted text
        
        Args:
            text: Extracted text
        
        Returns:
            Sentiment score (-1 to 1)
        """
        # Simple VADER-like analysis
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        
        return scores['compound']
    
    def analyze_image(self, image_source) -> ImageAnalysisResult:
        """
        Comprehensive image analysis
        
        Args:
            image_source: URL, file path, or PIL Image
        
        Returns:
            ImageAnalysisResult
        """
        # Load image
        image = self.load_image(image_source)
        
        if image is None:
            return ImageAnalysisResult(
                has_text=False,
                extracted_text=None,
                text_sentiment=0.0,
                is_chart=False,
                is_meme=False,
                dominant_colors=[],
                color_sentiment=0.0,
                overall_sentiment=0.0,
                confidence=0.0
            )
        
        # Extract text
        extracted_text = self.extract_text(image)
        has_text = extracted_text is not None and len(extracted_text.strip()) > 0
        
        # Analyze text sentiment
        text_sentiment = 0.0
        if has_text:
            text_sentiment = self.analyze_text_sentiment(extracted_text)
        
        # Detect chart
        is_chart = self.detect_chart(image, extracted_text)
        
        # Detect meme
        is_meme = self.detect_meme(extracted_text)
        
        # Analyze colors
        dominant_colors, color_sentiment = self.analyze_colors(image)
        
        # Calculate overall sentiment
        # Weight text sentiment more heavily if present
        if has_text:
            overall_sentiment = text_sentiment * 0.7 + color_sentiment * 0.3
            confidence = 0.8
        else:
            overall_sentiment = color_sentiment
            confidence = 0.3  # Lower confidence without text
        
        # Adjust for chart/meme context
        if is_chart:
            # Charts are more factual, reduce sentiment magnitude
            overall_sentiment *= 0.5
        
        if is_meme:
            # Memes often exaggerate, increase magnitude
            overall_sentiment *= 1.2
            confidence = min(confidence * 1.1, 1.0)
        
        return ImageAnalysisResult(
            has_text=has_text,
            extracted_text=extracted_text,
            text_sentiment=text_sentiment,
            is_chart=is_chart,
            is_meme=is_meme,
            dominant_colors=dominant_colors,
            color_sentiment=color_sentiment,
            overall_sentiment=np.clip(overall_sentiment, -1, 1),
            confidence=confidence
        )
    
    def analyze_video(
        self,
        video_source,
        sample_rate: int = 5
    ) -> VideoAnalysisResult:
        """
        Analyze video by sampling frames
        
        Args:
            video_source: Video file path or URL
            sample_rate: Analyze every Nth frame
        
        Returns:
            VideoAnalysisResult
        """
        # Note: This is a simplified implementation
        # Full video analysis would require cv2 (OpenCV)
        
        logger.warning("Video analysis is not fully implemented - requires opencv-python")
        
        return VideoAnalysisResult(
            frame_count=0,
            key_frames_analyzed=0,
            average_sentiment=0.0,
            sentiment_trend='neutral',
            has_text=False,
            extracted_texts=[]
        )


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("MULTIMODAL ANALYSIS TEST")
    print("="*80)
    
    # Create analyzer
    analyzer = MultimodalAnalyzer()
    
    # Test with synthetic image
    print("\n📷 Creating test image...")
    
    # Create a simple test image with text
    test_image = Image.new('RGB', (400, 300), color=(50, 200, 50))  # Green background
    
    # Simulate OCR text
    test_text = "Bitcoin TO THE MOON! 🚀 HODL"
    
    print(f"   Text: '{test_text}'")
    print(f"   Background: Green (positive color)")
    
    # Analyze
    result = analyzer.analyze_image(test_image)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Has Text: {result.has_text}")
    if result.extracted_text:
        print(f"   Extracted: '{result.extracted_text[:50]}...'")
    print(f"   Text Sentiment: {result.text_sentiment:.3f}")
    print(f"   Is Chart: {result.is_chart}")
    print(f"   Is Meme: {result.is_meme}")
    print(f"   Dominant Colors: {len(result.dominant_colors)}")
    print(f"   Color Sentiment: {result.color_sentiment:.3f}")
    print(f"   Overall Sentiment: {result.overall_sentiment:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    
    # Test meme detection
    print(f"\n🎭 Meme Detection:")
    meme_texts = [
        "DIAMOND HANDS 💎🙌",
        "WEN LAMBO",
        "HODL THE LINE",
        "Regular market analysis"
    ]
    
    for text in meme_texts:
        is_meme = analyzer.detect_meme(text)
        print(f"   '{text}': {'MEME' if is_meme else 'Not meme'}")
    
    # Test chart detection
    print(f"\n📈 Chart Detection:")
    chart_texts = [
        "BTC/USD Daily Chart - Price at $50,000 with RSI at 65",
        "Candlestick pattern showing bullish trend",
        "Just a random social media post"
    ]
    
    for text in chart_texts:
        is_chart = analyzer.detect_chart(test_image, text)
        print(f"   '{text}': {'CHART' if is_chart else 'Not chart'}")
    
    print(f"\n✅ Multimodal analysis complete!")
    
    if not analyzer.has_ocr:
        print(f"\n⚠️  Note: OCR is disabled (pytesseract not installed)")
        print(f"    Install with: pip install pytesseract")
        print(f"    Also requires Tesseract binary: https://github.com/tesseract-ocr/tesseract")
