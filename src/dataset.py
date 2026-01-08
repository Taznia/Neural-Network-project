import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def load_data(base_path):
    # Load audio features (GTZAN) - It's in data/audio/
    audio_path = os.path.join(base_path, 'data', 'audio', 'features_3_sec.csv')
    audio_df = pd.read_csv(audio_path)
    
    # Load English lyrics - It's in data/lyrics/
    eng_lyrics_path = os.path.join(base_path, 'data', 'lyrics', 'lyrics.csv')
    eng_df = pd.read_csv(eng_lyrics_path)
    
    # Load Bangla lyrics - It's in data/lyrics/
    ban_lyrics_path = os.path.join(base_path, 'data', 'lyrics', 'BanglaSongLyrics.csv')
    ban_df = pd.read_csv(ban_lyrics_path)
    
    return audio_df, eng_df, ban_df

def preprocess_lyrics(df, text_col='lyrics', max_features=100):
    # Basic cleaning and TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    # For Bangla, we might need different stop words, but for basic TF-IDF it's okay for now
    lyrics_features = tfidf.fit_transform(df[text_col].fillna('').astype(str)).toarray()
    return lyrics_features

def prepare_unified_dataset(audio_df, eng_df, ban_df, n_samples=3000):
    audio_cols = [c for c in audio_df.columns if c not in ['filename', 'length', 'label']]
    X_audio_raw = audio_df[audio_cols].values
    y_genre_raw = audio_df['label'].values
    
    # Encoder for genre labels
    unique_genres = np.sort(np.unique(y_genre_raw))
    genre_to_int = {g: i for i, g in enumerate(unique_genres)}
    y_genre_int = np.array([genre_to_int[g] for g in y_genre_raw])
    
    # Normalize Audio
    scaler_audio = StandardScaler()
    X_audio_scaled = scaler_audio.fit_transform(X_audio_raw)
    
    # Pooling English and Bangla lyrics to create a truly hybrid dataset
    eng_df['lang'] = 'English'
    ban_df['lang'] = 'Bangla'
    
    # Standardize column names for lyrics
    if 'lyrics' not in eng_df.columns and 'lyric' in eng_df.columns:
        eng_df = eng_df.rename(columns={'lyric': 'lyrics'})
    elif 'lyrics' not in eng_df.columns and 'text' in eng_df.columns:
        eng_df = eng_df.rename(columns={'text': 'lyrics'})

    combined_lyrics_df = pd.concat([
        eng_df[['lyrics', 'lang']], 
        ban_df.rename(columns={'Lyric': 'lyrics', 'lyrics': 'lyrics'})[['lyrics', 'lang']]
    ], ignore_index=True).dropna(subset=['lyrics'])

    total_samples = min(n_samples, len(X_audio_raw), len(combined_lyrics_df))
    
    # Sample subsets
    idx_audio = np.random.choice(len(X_audio_raw), total_samples, replace=False)
    idx_lyrics = np.random.choice(len(combined_lyrics_df), total_samples, replace=False)
    
    X_audio_sub = X_audio_scaled[idx_audio]
    y_genre_sub = y_genre_int[idx_audio]
    
    # Lyrics features (Hybrid English + Bangla)
    lyrics_sub = combined_lyrics_df.iloc[idx_lyrics]
    X_lyrics_sub = preprocess_lyrics(lyrics_sub, 'lyrics', max_features=128)
    
    # Language labels for verification
    lang_labels = lyrics_sub['lang'].values
    
    # PCA to common dimension
    common_dim = 32
    pca_audio = PCA(n_components=common_dim)
    X_audio_proj = pca_audio.fit_transform(X_audio_sub)
    
    pca_lyrics = PCA(n_components=common_dim)
    X_lyrics_proj = pca_lyrics.fit_transform(X_lyrics_sub)
    
    return X_audio_proj, X_lyrics_proj, y_genre_sub, unique_genres, lang_labels

if __name__ == "__main__":
    # Get the directory where this script is located
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_df, eng_df, ban_df = load_data(base_path)
    X_a_proj, X_l_proj, y_genre, genre_names, lang_labels = prepare_unified_dataset(audio_df, eng_df, ban_df)
    print(f"Paired Dataset Created:")
    print(f"Audio Features: {X_a_proj.shape}")
    print(f"Lyrics Features: {X_l_proj.shape}")
    print(f"Genre labels: {y_genre.shape}")
    print(f"Number of Genres: {len(genre_names)}")
    print(f"Languages: {np.unique(lang_labels)}")

