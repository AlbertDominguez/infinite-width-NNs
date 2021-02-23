import neural_tangents as nt
from jax.api import jit
from neural_tangents import stax

def define(args):
    hidden_layers = []
    for _ in range(args.num_hidden_layers):
        hidden_layers.append(stax.Dense(args.hidden_neurons, W_std=args.W_std, b_std=args.b_std))
        hidden_layers.append(stax.Relu())
    init_fn, apply_fn, kernel_fn = stax.serial(
        *hidden_layers,
        stax.Dense(args.output_dim, W_std=args.W_std, b_std=args.b_std)
    )
    apply_fn = jit(apply_fn)
    batched_kernel_fn = nt.batch(kernel_fn, batch_size=args.batch_size, device_count=-1)
    return init_fn, apply_fn, kernel_fn, batched_kernel_fn
