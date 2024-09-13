import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def inspect_chunks(
    pages_and_chunks: list[dict], 
    nr_rows: int = 5
):
    """
    Inspects and visualizes the distribution of character, word, and token counts in text chunks.

    Args:
        pages_and_chunks (list[dict]): A list of dictionaries, each containing information about a chunk of text.
                                       The dictionary keys should include:
                                       - 'file_name': The name of the file.
                                       - 'page_number': The page number (starting from 1).
                                       - 'chunk_text': The text of the chunk.
                                       - 'chunk_char_count': The number of characters in the chunk.
                                       - 'chunk_word_count': The number of words in the chunk.
                                       - 'chunk_token_count': The estimated number of tokens in the chunk.
        nr_rows (int): The number of top and bottom rows to display based on the character count. Default is 5.

    Returns:
        pd.DataFrame: A DataFrame containing the statistics of the chunk text, including character, word, and token counts.
    """
    df = pd.DataFrame(pages_and_chunks)

    stats = df[['chunk_char_count', 'chunk_word_count', 'chunk_token_count']].describe()
    print("Descriptive statistics for chunk counts:")
    print(stats)

    # Sort by chunk_char_count in descending order and display the top rows
    head_char_count = df.sort_values(by='chunk_char_count', ascending=False).head(nr_rows)
    print(f"\nTop {nr_rows} chunks by character count:")
    print(head_char_count[['file_name', 'page_number', 'chunk_text', 'chunk_char_count']])

    # Sort by chunk_char_count in descending order and display the last rows
    tail_char_count = df.sort_values(by='chunk_char_count', ascending=False).tail(nr_rows)
    print(f"\nLast {nr_rows} chunks by character count:")
    print(tail_char_count[['file_name', 'page_number', 'chunk_text', 'chunk_char_count']])

    # Plot distributions
    plt.figure(figsize=(12, 6))

    # Plot chunk_char_count
    plt.subplot(1, 3, 1)
    sns.histplot(df['chunk_char_count'], kde=True)
    plt.title('Character Count Distribution')

    # Plot chunk_word_count
    plt.subplot(1, 3, 2)
    sns.histplot(df['chunk_word_count'], kde=True)
    plt.title('Word Count Distribution')

    # Plot chunk_token_count
    plt.subplot(1, 3, 3)
    sns.histplot(df['chunk_token_count'], kde=True)
    plt.title('Token Count Distribution')

    plt.tight_layout()
    plt.show()

    return stats
