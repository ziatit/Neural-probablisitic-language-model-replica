def create_model(vocab_size, n_emb, n_hidden, context_window, device):
    """Create a Bengio-style neural language model.
    
    Args:
        vocab_size: Size of the vocabulary
        n_emb: Embedding dimension
        n_hidden: Hidden layer dimension
        context_window: Number of context words (block_size)
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Sequential model
        n_params: Number of trainable parameters
    """
    model = Sequential([
        Embeddings(vocab_size, n_emb, device=device),
        Linear(n_emb * context_window, n_hidden, bias=True, device=device),
        Tanh(),
        Linear(n_hidden, vocab_size, bias=True, device=device),
    ])
    
    parameters = model.parameters()
    n_params = sum(p.nelement() for p in parameters)
    for p in parameters:
        p.requires_grad = True
    
    return model, n_params