"""
Crowd Psychology Analysis

Analyzes collective behavior patterns in financial markets:
- Social contagion (information spreading)
- Echo chambers (filter bubbles)
- Influencer impact (key opinion leaders)
- Sentiment cascades (viral sentiment shifts)
- Network effects (community structure)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from loguru import logger


@dataclass
class InfluencerProfile:
    """Profile of an influential user"""
    user_id: str
    username: str
    follower_count: int
    engagement_rate: float
    average_sentiment: float
    post_frequency: float  # posts per day
    influence_score: float  # 0-1


@dataclass
class EchoChamber:
    """Detected echo chamber (filter bubble)"""
    chamber_id: int
    members: Set[str]
    dominant_sentiment: str  # 'positive', 'negative', 'neutral'
    sentiment_homogeneity: float  # 0-1, higher = more echo chamber-like
    average_engagement: float
    common_hashtags: List[str]


@dataclass
class SentimentCascade:
    """Viral sentiment cascade event"""
    cascade_id: int
    start_time: datetime
    end_time: datetime
    initial_sentiment: float
    peak_sentiment: float
    affected_users: int
    spread_rate: float  # users per hour
    trigger_content: Optional[str]


@dataclass
class CrowdPsychologyMetrics:
    """Aggregate crowd psychology metrics"""
    timestamp: datetime
    herd_mentality_score: float  # 0-1, alignment of opinions
    echo_chamber_index: float    # 0-1, degree of fragmentation
    influencer_impact: float     # 0-1, concentration of influence
    sentiment_volatility: float  # std of sentiment changes
    contagion_rate: float       # speed of sentiment spread
    network_polarization: float  # 0-1, opinion polarization


class CrowdPsychologyAnalyzer:
    """
    Analyze crowd psychology in social media data
    
    Features:
    - Identify influential users
    - Detect echo chambers
    - Track sentiment cascades
    - Measure herd mentality
    - Analyze network effects
    """
    
    def __init__(
        self,
        influencer_threshold: int = 1000,
        echo_chamber_threshold: float = 0.7
    ):
        """
        Initialize crowd psychology analyzer
        
        Args:
            influencer_threshold: Min followers to be considered influencer
            echo_chamber_threshold: Similarity threshold for echo chambers
        """
        self.influencer_threshold = influencer_threshold
        self.echo_chamber_threshold = echo_chamber_threshold
        
        self.influencers: Dict[str, InfluencerProfile] = {}
        self.echo_chambers: List[EchoChamber] = []
        self.sentiment_cascades: List[SentimentCascade] = []
        
        logger.info("CrowdPsychologyAnalyzer initialized")
    
    def identify_influencers(
        self,
        social_data: pd.DataFrame
    ) -> List[InfluencerProfile]:
        """
        Identify influential users
        
        Args:
            social_data: DataFrame with columns:
                - user_id, username, follower_count, timestamp
                - sentiment_score, engagement (likes, retweets, etc.)
        
        Returns:
            List of InfluencerProfile objects
        """
        if 'user_id' not in social_data.columns:
            logger.warning("No user_id column in data")
            return []
        
        influencers = []
        
        # Group by user
        user_groups = social_data.groupby('user_id')
        
        for user_id, user_data in user_groups:
            # Get user info
            username = user_data['username'].iloc[0] if 'username' in user_data else user_id
            follower_count = user_data['follower_count'].iloc[0] if 'follower_count' in user_data else 0
            
            if follower_count < self.influencer_threshold:
                continue
            
            # Calculate metrics
            avg_sentiment = user_data['sentiment_score'].mean() if 'sentiment_score' in user_data else 0.0
            
            # Engagement rate
            if 'engagement' in user_data.columns:
                engagement_rate = user_data['engagement'].mean() / max(follower_count, 1)
            else:
                engagement_rate = 0.0
            
            # Post frequency
            time_span = (user_data['timestamp'].max() - user_data['timestamp'].min()).days
            post_frequency = len(user_data) / max(time_span, 1)
            
            # Influence score (combination of followers, engagement, activity)
            influence_score = min(
                (np.log10(follower_count + 1) / 7.0) * 0.5 +  # Follower influence (max ~10M followers)
                engagement_rate * 0.3 +
                min(post_frequency / 10, 1.0) * 0.2,  # Activity
                1.0
            )
            
            influencer = InfluencerProfile(
                user_id=user_id,
                username=username,
                follower_count=follower_count,
                engagement_rate=engagement_rate,
                average_sentiment=avg_sentiment,
                post_frequency=post_frequency,
                influence_score=influence_score
            )
            
            influencers.append(influencer)
        
        # Sort by influence score
        influencers.sort(key=lambda x: x.influence_score, reverse=True)
        
        self.influencers = {inf.user_id: inf for inf in influencers}
        
        logger.info(f"Identified {len(influencers)} influencers")
        return influencers
    
    def detect_echo_chambers(
        self,
        social_data: pd.DataFrame,
        min_chamber_size: int = 5
    ) -> List[EchoChamber]:
        """
        Detect echo chambers (groups with homogeneous sentiment)
        
        Args:
            social_data: Social media data
            min_chamber_size: Minimum users in a chamber
        
        Returns:
            List of EchoChamber objects
        """
        if 'user_id' not in social_data.columns or 'sentiment_score' not in social_data.columns:
            logger.warning("Missing required columns for echo chamber detection")
            return []
        
        # Calculate user sentiment profiles
        user_profiles = social_data.groupby('user_id').agg({
            'sentiment_score': ['mean', 'std'],
            'engagement': 'mean' if 'engagement' in social_data.columns else 'count'
        }).reset_index()
        
        user_profiles.columns = ['user_id', 'avg_sentiment', 'sentiment_std', 'avg_engagement']
        
        # Simple clustering based on sentiment similarity
        from sklearn.cluster import DBSCAN
        
        # Cluster users by sentiment
        X = user_profiles[['avg_sentiment']].values
        clustering = DBSCAN(eps=0.2, min_samples=min_chamber_size)
        clusters = clustering.fit_predict(X)
        
        # Create echo chambers
        echo_chambers = []
        chamber_id = 0
        
        for cluster_label in set(clusters):
            if cluster_label == -1:  # Noise points
                continue
            
            cluster_mask = clusters == cluster_label
            chamber_users = set(user_profiles.loc[cluster_mask, 'user_id'])
            
            if len(chamber_users) < min_chamber_size:
                continue
            
            # Calculate chamber metrics
            chamber_data = user_profiles.loc[cluster_mask]
            avg_sentiment = chamber_data['avg_sentiment'].mean()
            
            # Sentiment homogeneity (inverse of std dev)
            sentiment_std = chamber_data['avg_sentiment'].std()
            homogeneity = 1.0 / (1.0 + sentiment_std)
            
            # Dominant sentiment
            if avg_sentiment > 0.2:
                dominant = 'positive'
            elif avg_sentiment < -0.2:
                dominant = 'negative'
            else:
                dominant = 'neutral'
            
            avg_engagement = chamber_data['avg_engagement'].mean()
            
            # Get common hashtags
            chamber_posts = social_data[social_data['user_id'].isin(chamber_users)]
            if 'hashtags' in chamber_posts.columns:
                all_hashtags = []
                for tags in chamber_posts['hashtags'].dropna():
                    if isinstance(tags, list):
                        all_hashtags.extend(tags)
                common_hashtags = [tag for tag, _ in Counter(all_hashtags).most_common(5)]
            else:
                common_hashtags = []
            
            chamber = EchoChamber(
                chamber_id=chamber_id,
                members=chamber_users,
                dominant_sentiment=dominant,
                sentiment_homogeneity=homogeneity,
                average_engagement=avg_engagement,
                common_hashtags=common_hashtags
            )
            
            echo_chambers.append(chamber)
            chamber_id += 1
        
        self.echo_chambers = echo_chambers
        
        logger.info(f"Detected {len(echo_chambers)} echo chambers")
        return echo_chambers
    
    def detect_sentiment_cascades(
        self,
        social_data: pd.DataFrame,
        time_window: int = 6  # hours
    ) -> List[SentimentCascade]:
        """
        Detect viral sentiment cascades
        
        A cascade is a rapid spread of sentiment change
        
        Args:
            social_data: Social media data with timestamps
            time_window: Time window in hours to detect cascades
        
        Returns:
            List of SentimentCascade objects
        """
        if 'timestamp' not in social_data.columns or 'sentiment_score' not in social_data.columns:
            logger.warning("Missing required columns for cascade detection")
            return []
        
        # Sort by timestamp
        data = social_data.sort_values('timestamp')
        
        # Calculate rolling sentiment
        data['rolling_sentiment'] = data['sentiment_score'].rolling(window=20, min_periods=1).mean()
        data['sentiment_change'] = data['rolling_sentiment'].diff()
        
        cascades = []
        cascade_id = 0
        
        # Detect rapid sentiment changes
        threshold = data['sentiment_change'].std() * 2  # 2 sigma events
        
        i = 0
        while i < len(data):
            if abs(data.iloc[i]['sentiment_change']) > threshold:
                # Potential cascade start
                cascade_start = data.iloc[i]['timestamp']
                initial_sentiment = data.iloc[i]['rolling_sentiment']
                
                # Find end of cascade (sentiment stabilizes)
                j = i + 1
                peak_sentiment = initial_sentiment
                affected_users = set([data.iloc[i].get('user_id', 'unknown')])
                
                while j < len(data):
                    time_diff = (data.iloc[j]['timestamp'] - cascade_start).total_seconds() / 3600
                    
                    if time_diff > time_window:
                        break
                    
                    current_sentiment = data.iloc[j]['rolling_sentiment']
                    if abs(current_sentiment) > abs(peak_sentiment):
                        peak_sentiment = current_sentiment
                    
                    if 'user_id' in data.columns:
                        affected_users.add(data.iloc[j]['user_id'])
                    
                    j += 1
                
                cascade_end = data.iloc[j-1]['timestamp'] if j > i else cascade_start
                duration_hours = (cascade_end - cascade_start).total_seconds() / 3600
                spread_rate = len(affected_users) / max(duration_hours, 0.1)
                
                # Only record significant cascades
                if abs(peak_sentiment - initial_sentiment) > 0.3 and len(affected_users) > 5:
                    trigger = data.iloc[i].get('text', None)
                    
                    cascade = SentimentCascade(
                        cascade_id=cascade_id,
                        start_time=cascade_start,
                        end_time=cascade_end,
                        initial_sentiment=initial_sentiment,
                        peak_sentiment=peak_sentiment,
                        affected_users=len(affected_users),
                        spread_rate=spread_rate,
                        trigger_content=trigger[:100] if trigger else None
                    )
                    
                    cascades.append(cascade)
                    cascade_id += 1
                
                i = j
            else:
                i += 1
        
        self.sentiment_cascades = cascades
        
        logger.info(f"Detected {len(cascades)} sentiment cascades")
        return cascades
    
    def calculate_herd_mentality(self, social_data: pd.DataFrame) -> float:
        """
        Calculate herd mentality score
        
        High score = high alignment of opinions (everyone thinks alike)
        
        Args:
            social_data: Social media data
        
        Returns:
            Herd mentality score (0-1)
        """
        if 'sentiment_score' not in social_data.columns or len(social_data) < 2:
            return 0.0
        
        # Calculate sentiment variance
        sentiment_std = social_data['sentiment_score'].std()
        
        # Low variance = high herd mentality
        herd_score = 1.0 / (1.0 + sentiment_std)
        
        return min(herd_score, 1.0)
    
    def calculate_polarization(self, social_data: pd.DataFrame) -> float:
        """
        Calculate network polarization
        
        High polarization = opinions split into opposing camps
        
        Args:
            social_data: Social media data
        
        Returns:
            Polarization score (0-1)
        """
        if 'sentiment_score' not in social_data.columns or len(social_data) < 2:
            return 0.0
        
        sentiments = social_data['sentiment_score'].values
        
        # Count positive and negative
        positive = (sentiments > 0.2).sum()
        negative = (sentiments < -0.2).sum()
        neutral = ((sentiments >= -0.2) & (sentiments <= 0.2)).sum()
        
        total = len(sentiments)
        
        # Polarization is high when opinions are split (few neutrals)
        if total == 0:
            return 0.0
        
        # Bimodality coefficient
        extremes = (positive + negative) / total
        polarization = extremes * (1.0 - neutral / total)
        
        return min(polarization, 1.0)
    
    def analyze(
        self,
        social_data: pd.DataFrame,
        time_window: Optional[datetime] = None
    ) -> CrowdPsychologyMetrics:
        """
        Comprehensive crowd psychology analysis
        
        Args:
            social_data: Social media data
            time_window: Optional specific timestamp for metrics
        
        Returns:
            CrowdPsychologyMetrics
        """
        timestamp = time_window or datetime.now()
        
        # Identify influencers
        influencers = self.identify_influencers(social_data)
        
        # Detect echo chambers
        echo_chambers = self.detect_echo_chambers(social_data)
        
        # Detect cascades
        cascades = self.detect_sentiment_cascades(social_data)
        
        # Calculate metrics
        herd_mentality = self.calculate_herd_mentality(social_data)
        polarization = self.calculate_polarization(social_data)
        
        # Echo chamber index
        if len(social_data) > 0 and 'user_id' in social_data.columns:
            unique_users = social_data['user_id'].nunique()
            users_in_chambers = sum(len(chamber.members) for chamber in echo_chambers)
            echo_chamber_index = users_in_chambers / max(unique_users, 1)
        else:
            echo_chamber_index = 0.0
        
        # Influencer impact
        if influencers and 'user_id' in social_data.columns:
            influencer_posts = social_data[social_data['user_id'].isin(self.influencers.keys())]
            influencer_impact = len(influencer_posts) / max(len(social_data), 1)
        else:
            influencer_impact = 0.0
        
        # Sentiment volatility
        if 'sentiment_score' in social_data.columns:
            sentiment_changes = social_data['sentiment_score'].diff().dropna()
            sentiment_volatility = sentiment_changes.std() if len(sentiment_changes) > 0 else 0.0
        else:
            sentiment_volatility = 0.0
        
        # Contagion rate (from cascades)
        if cascades:
            avg_spread_rate = np.mean([c.spread_rate for c in cascades])
            contagion_rate = min(avg_spread_rate / 100, 1.0)  # Normalize
        else:
            contagion_rate = 0.0
        
        metrics = CrowdPsychologyMetrics(
            timestamp=timestamp,
            herd_mentality_score=herd_mentality,
            echo_chamber_index=echo_chamber_index,
            influencer_impact=influencer_impact,
            sentiment_volatility=sentiment_volatility,
            contagion_rate=contagion_rate,
            network_polarization=polarization
        )
        
        return metrics


# Test function
if __name__ == "__main__":
    # Generate sample social media data
    np.random.seed(42)
    
    num_users = 100
    num_posts = 500
    
    data = []
    for i in range(num_posts):
        user_id = f"user_{np.random.randint(0, num_users)}"
        
        # Simulate some influencers
        if int(user_id.split('_')[1]) < 10:
            follower_count = np.random.randint(10000, 100000)
            engagement = np.random.randint(100, 1000)
        else:
            follower_count = np.random.randint(100, 5000)
            engagement = np.random.randint(10, 100)
        
        # Create sentiment with some echo chambers
        if int(user_id.split('_')[1]) < 30:
            sentiment = np.random.normal(0.6, 0.1)  # Positive echo chamber
        elif int(user_id.split('_')[1]) < 60:
            sentiment = np.random.normal(-0.5, 0.15)  # Negative echo chamber
        else:
            sentiment = np.random.normal(0, 0.3)  # Mixed
        
        data.append({
            'user_id': user_id,
            'username': f"@{user_id}",
            'follower_count': follower_count,
            'sentiment_score': np.clip(sentiment, -1, 1),
            'engagement': engagement,
            'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 48)),
            'text': f"Sample post {i}"
        })
    
    social_data = pd.DataFrame(data)
    
    # Analyze
    analyzer = CrowdPsychologyAnalyzer()
    
    print("\n" + "="*80)
    print("CROWD PSYCHOLOGY ANALYSIS")
    print("="*80)
    
    metrics = analyzer.analyze(social_data)
    
    print(f"\nTimestamp: {metrics.timestamp}")
    print(f"Herd Mentality: {metrics.herd_mentality_score:.3f}")
    print(f"Echo Chamber Index: {metrics.echo_chamber_index:.3f}")
    print(f"Influencer Impact: {metrics.influencer_impact:.3f}")
    print(f"Sentiment Volatility: {metrics.sentiment_volatility:.3f}")
    print(f"Contagion Rate: {metrics.contagion_rate:.3f}")
    print(f"Network Polarization: {metrics.network_polarization:.3f}")
    
    print(f"\n🎯 Top Influencers:")
    for inf in list(analyzer.influencers.values())[:5]:
        print(f"  {inf.username}: {inf.follower_count:,} followers, "
              f"influence score: {inf.influence_score:.3f}")
    
    print(f"\n🔄 Echo Chambers: {len(analyzer.echo_chambers)}")
    for chamber in analyzer.echo_chambers:
        print(f"  Chamber {chamber.chamber_id}: {len(chamber.members)} members, "
              f"{chamber.dominant_sentiment} sentiment "
              f"(homogeneity: {chamber.sentiment_homogeneity:.3f})")
    
    print(f"\n⚡ Sentiment Cascades: {len(analyzer.sentiment_cascades)}")
    for cascade in analyzer.sentiment_cascades:
        print(f"  Cascade {cascade.cascade_id}: {cascade.affected_users} users affected, "
              f"spread rate: {cascade.spread_rate:.1f} users/hour")
