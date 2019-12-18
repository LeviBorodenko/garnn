__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


def is_using_reverse_process(input_shape):
    """Check if output of attention meachanism is a single
    Attention matrix or 2 attention matrices - one for A_in
    one for A_out

    [description]

    Arguments:
        input_shape {[tuple]} -- input_shape
    """

    # dimension of attention layer output
    dim = len(input_shape)

    # (batch, 2, N, N) if we use A_in and A_out
    if dim == 4:
        return True

    # (batch, N, N) is we aren't
    elif dim == 3:
        return False
    else:
        raise ValueError(f"Invalid attention shape {input_shape}")
