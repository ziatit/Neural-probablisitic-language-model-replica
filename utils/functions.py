import torch
import torch.nn.functional as F
import math
from utils.data_preparator import *
from utils.functions import *
from abstractions.classes import * 
import nltk 
import mlflow
import mlflow
nltk.download('brown')
from nltk.corpus import brown

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.synchronize()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')


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
        FlattenConsecutive(context_window),
        Linear(n_emb * context_window, n_hidden, bias=True, device=device),
        Tanh(),
        Linear(n_hidden, vocab_size, bias=True, device=device),
    ])
    
    parameters = model.parameters()
    n_params = sum(p.nelement() for p in parameters)
    for p in parameters:
        p.requires_grad = True
    
    return model, n_params

@torch.no_grad()
def evaluate_loss(model, X, Y, batch_size=256):
    """Evaluate loss on a dataset.
    
    Args:
        model: The language model
        X: Input tensor
        Y: Target tensor
        batch_size: Batch size for evaluation
    
    Returns:
        loss: Mean cross-entropy loss
    """
    losses = []
    for i in range(0, X.shape[0], batch_size):
        Xb = X[i:i + batch_size]
        Yb = Y[i:i + batch_size]
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        losses.append(loss.item())
    return sum(losses) / len(losses)

def train_experiment(params):
    """Run a training experiment with the given parameters.
    
    Args:
        params: Dictionary containing:
            - embedding_dim: Dimension of word embeddings
            - hidden_dim: Dimension of hidden layer
            - learning_rate: Learning rate
            - batch_size: Training batch size
            - context_window: Number of context words
            - max_steps: Maximum training steps
            - eval_interval: Evaluate every N steps
            - lr_decay_step: Step at which to decay learning rate (optional)
            - seed: Random seed (optional)
    
    Returns:
        dict: Results including final train_loss, val_loss, and perplexity
    """
    # Extract parameters with defaults
    n_emb = params.get('embedding_dim', 60)
    n_hidden = params.get('hidden_dim', 100)
    learning_rate = params.get('learning_rate', 0.1)
    batch_size = params.get('batch_size', 64)
    context_window = params.get('context_window', 3)
    max_steps = params.get('max_steps', 10000)
    eval_interval = params.get('eval_interval', 1000)
    lr_decay_step = params.get('lr_decay_step', 150000)
    seed = params.get('seed', 42)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Build dataset
    Xtr, Xval, Xte, Ytr, Yval, Yte, vocab_size, stoi, itos = build_dataset(
        brown.words(), context_window, device=device
    )
    
    # Create model
    model, n_params = create_model(vocab_size, n_emb, n_hidden, context_window, device)
    parameters = model.parameters()
    
    print(f'\n{"="*60}')
    print(f'Starting experiment: n_emb={n_emb}, n_hidden={n_hidden}')
    print(f'Model parameters: {n_params:,}')
    print(f'Vocab size: {vocab_size}')
    print(f'{"="*60}\n')
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'embedding_dim': n_emb,
            'hidden_dim': n_hidden,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'context_window': context_window,
            'max_steps': max_steps,
            'vocab_size': vocab_size,
            'n_params': n_params,
            'seed': seed,
            'device': device
        })
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for step in range(max_steps):
            # Sample batch
            ix = torch.randint(0, Xtr.shape[0], (batch_size,))
            Xb, Yb = Xtr[ix], Ytr[ix]
            
            # Forward pass
            logits = model(Xb)
            loss = F.cross_entropy(logits, Yb)
            
            # Backward pass
            for p in parameters:
                p.grad = None
            loss.backward()
            
            # Learning rate schedule
            lr = learning_rate if step < lr_decay_step else learning_rate * 0.1
            
            # Update parameters
            for p in parameters:
                p.data += -lr * p.grad
            
            # Log and evaluate
            if step % eval_interval == 0 or step == max_steps - 1:
                train_loss = evaluate_loss(model, Xtr, Ytr)
                val_loss = evaluate_loss(model, Xval, Yval)
                train_perplexity = math.exp(train_loss)
                val_perplexity = math.exp(val_loss)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_perplexity': train_perplexity,
                    'val_perplexity': val_perplexity
                }, step=step)
                
                print(f'Step {step:6d}/{max_steps}: '
                      f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
                      f'train_ppl={train_perplexity:.2f}, val_ppl={val_perplexity:.2f}')
        
        # Final evaluation
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        final_train_ppl = math.exp(final_train_loss)
        final_val_ppl = math.exp(final_val_loss)
        test_loss = evaluate_loss(model, Xte, Yte)
        test_perplexity = math.exp(test_loss)
        
        # Log final metrics
        mlflow.log_metrics({
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_test_loss': test_loss,
            'final_train_perplexity': final_train_ppl,
            'final_val_perplexity': final_val_ppl,
            'final_test_perplexity': test_perplexity
        })
        
        print(f'\nFinal Results:')
        print(f'  Train Loss: {final_train_loss:.4f}, Perplexity: {final_train_ppl:.2f}')
        print(f'  Val Loss:   {final_val_loss:.4f}, Perplexity: {final_val_ppl:.2f}')
        print(f'  Test Loss:  {test_loss:.4f}, Perplexity: {test_perplexity:.2f}')
        
        # Log number of parameters as metric for easy comparison
        mlflow.log_metric('n_params', n_params)
        
        run_id = mlflow.active_run().info.run_id
    
    return {
        'run_id': run_id,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'test_loss': test_loss,
        'train_perplexity': final_train_ppl,
        'val_perplexity': final_val_ppl,
        'test_perplexity': test_perplexity,
        'n_params': n_params,
        'model': model,
        'itos': itos,
        'stoi': stoi
    }
     