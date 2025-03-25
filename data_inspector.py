import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def inspect_chunks(chunk_data, nr_rows=5):
    """
    Inspects and visualizes the distribution of word counts in text chunks 
    (just an example for debugging or exploring chunk sizes).
    
    Args:
        chunk_data (list[dict]): Each dictionary typically has:
            - 'file_name'
            - 'page_number'
            - 'chunk_text'
            - 'chunk_char_count'
            - 'chunk_word_count'
        nr_rows (int): How many top or bottom rows to display based on char count.

    Returns:
        pd.DataFrame: A DataFrame describing basic stats of the chunk text.
    """
    df = pd.DataFrame(chunk_data)
    if "chunk_word_count" not in df.columns:
        print("No 'chunk_word_count' in the data. Nothing to inspect.")
        return None

    # Basic stats
    stats = df["chunk_word_count"].describe()
    print("Descriptive statistics for word counts:")
    print(stats)

    # Sort by chunk_char_count descending
    head_char_count = df.sort_values(by='chunk_char_count', ascending=False).head(nr_rows)
    print(f"\nTop {nr_rows} chunks by character count:")
    print(head_char_count[['file_name', 'page_number', 'chunk_text', 'chunk_char_count']])

    tail_char_count = df.sort_values(by='chunk_char_count', ascending=False).tail(nr_rows)
    print(f"\nLast {nr_rows} chunks by character count:")
    print(tail_char_count[['file_name', 'page_number', 'chunk_text', 'chunk_char_count']])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df['chunk_char_count'], kde=True)
    plt.title('Character Count Distribution')

    plt.subplot(1, 2, 2)
    sns.histplot(df['chunk_word_count'], kde=True)
    plt.title('Word Count Distribution')

    plt.tight_layout()
    plt.show()

    return df
