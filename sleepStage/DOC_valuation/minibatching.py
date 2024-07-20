import math
import numpy as np


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Generate a generator that return a batch of inputs and targets.
    """

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def iterate_batch_seq_minibatches(eeginputs, eoginputs, targets, sleepinputs, batch_size, seq_length):
    """
    Generate a generator that return a batch of sequences of inputs and targets.

    This function splits a sequence of inputs and targets into multiple sub-
    sequences equally. Then it further splits each sub-sequence into multiple
    chunks with the size of seq_length.
    """

    assert len(eeginputs) == len(targets) == len(eoginputs)
    n_inputs = len(eeginputs)
    batch_len = n_inputs // batch_size

    epoch_size = batch_len // seq_length
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    seq_eeginputs = np.zeros((batch_size, batch_len) + eeginputs.shape[1:],
                          dtype=eeginputs.dtype)
    seq_eoginputs = np.zeros((batch_size, batch_len) + eoginputs.shape[1:],
                          dtype=eoginputs.dtype)
    seq_sleepinputs = np.zeros((batch_size, batch_len) + sleepinputs.shape[1:],
                          dtype=sleepinputs.dtype)  # 新增睡眠分期结果
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:],
                           dtype=targets.dtype)

    for i in range(batch_size):
        seq_eeginputs[i] = eeginputs[i*batch_len:(i+1)*batch_len]
        seq_eoginputs[i] = eoginputs[i*batch_len:(i+1)*batch_len]
        seq_sleepinputs[i] = sleepinputs[i * batch_len:(i + 1) * batch_len]

        seq_targets[i] = targets[i * batch_len:(i + 1) * batch_len]

    for i in range(epoch_size):
        x = seq_eeginputs[:, i*seq_length:(i+1)*seq_length]
        y = seq_eoginputs[:, i*seq_length:(i+1)*seq_length]
        z = seq_targets[:, i*seq_length:(i+1)*seq_length]
        s = seq_sleepinputs[:, i*seq_length:(i+1)*seq_length]

        flatten_x = x.reshape((-1,) + eeginputs.shape[1:])
        flatten_y = y.reshape((-1,) + eoginputs.shape[1:])
        flatten_s = s.reshape((-1,) + sleepinputs.shape[1:])
        flatten_z = z.reshape((-1,) + targets.shape[1:])

        yield flatten_x, flatten_y, flatten_z, flatten_s


def iterate_batch_multiple_seq_minibatches(eeginputs, eoginputs, sleepinputs, targets, batch_size, seq_length, shuffle_idx=None, augment_seq=False):
    """
    Generate a generator that return a batch of sequences of inputs and targets.

    This function randomly selects batches of multiple sequence. It then iterates
    through multiple sequence in parallel to generate a sequence of inputs and
    targets. It will append the input sequence with 0 and target with -1 when
    the lenght of each sequence is not equal.
    """

    # assert len(eeginputs) == len(targets) == len(eoginputs)
    assert len(eeginputs) == len(targets) == len(eoginputs) == len(sleepinputs)     # 新增睡眠分期结果
    n_inputs = len(eeginputs)

    if shuffle_idx is None:
        # No shuffle
        seq_idx = np.arange(n_inputs)
    else:
        # Shuffle subjects (get the shuffled indices from argument)
        seq_idx = shuffle_idx

    eeginput_sample_shape = eeginputs[0].shape[1:]
    eoginput_sample_shape = eoginputs[0].shape[1:]
    sleepinput_sample_shape = sleepinputs[0].shape[1:]    # 新增睡眠分期结果
    target_sample_shape = targets[0].shape[1:]

    # Compute the number of maximum loops
    n_loops = int(math.ceil(len(seq_idx) / batch_size))

    # For each batch of subjects (size=batch_size)
    for l in range(n_loops):
        start_idx = l*batch_size
        end_idx = (l+1)*batch_size

        # Convert seq_idx to an integer array if it's not already
        seq_idx_slice = np.asarray(seq_idx[start_idx:end_idx], dtype=int)

        # seq_eeginputs = np.asarray(eeginputs)[seq_idx[start_idx:end_idx]]
        # seq_eoginputs = np.asarray(eoginputs)[seq_idx[start_idx:end_idx]]
        # seq_targets = np.asarray(targets)[seq_idx[start_idx:end_idx]]

        # Slice the original arrays using the integer array
        seq_eeginputs = np.asarray(eeginputs)[seq_idx_slice]
        seq_eoginputs = np.asarray(eoginputs)[seq_idx_slice]
        seq_sleepinputs = np.asarray(sleepinputs)[seq_idx_slice]    # 新增睡眠分期结果

        seq_targets = np.asarray(targets)[seq_idx_slice]

        if augment_seq:
            # Data augmentation: multiple sequences
            # Randomly skip some epochs at the beginning -> generate multiple sequence
            max_skips = 5
            # for s_idx in range(len(seq_eeginputs)): 修改了这里
            #     n_skips = np.random.randint(max_skips)
            #     if n_skips > 0 and n_skips < len(seq_eeginputs[s_idx]):
            #         seq_eeginputs[s_idx] = seq_eeginputs[s_idx][n_skips:]
            #         seq_eoginputs[s_idx] = seq_eoginputs[s_idx][n_skips:]
            #         seq_targets[s_idx] = seq_targets[s_idx][n_skips:]
            augmented_seq_eeginputs = []
            augmented_seq_eoginputs = []
            augmented_seq_sleepinputs = []     # 新增睡眠分期结果
            augmented_seq_targets = []

            for s_idx in range(len(seq_eeginputs)):
                n_skips = np.random.randint(max_skips)
                if n_skips > 0 and n_skips < len(seq_eeginputs[s_idx]):
                    augmented_seq_eeginputs.append(seq_eeginputs[s_idx][n_skips:])
                    augmented_seq_eoginputs.append(seq_eoginputs[s_idx][n_skips:])
                    augmented_seq_sleepinputs.append(seq_sleepinputs[s_idx][n_skips:])     # 新增睡眠分期结果
                    augmented_seq_targets.append(seq_targets[s_idx][n_skips:])

            # Convert the augmented sequences back to numpy arrays
            seq_eeginputs = np.asarray(augmented_seq_eeginputs)
            seq_eoginputs = np.asarray(augmented_seq_eoginputs)
            seq_sleepinputs = np.asarray(augmented_seq_sleepinputs)     # 新增睡眠分期结果
            seq_targets = np.asarray(augmented_seq_targets)


        # Determine the maximum number of batch sequences
        n_max_seq_inputs = -1
        for s_idx, s in enumerate(seq_eeginputs):
            if len(s) > n_max_seq_inputs:
                n_max_seq_inputs = len(s)
        for s_idx, s in enumerate(seq_eoginputs):
            if len(s) > n_max_seq_inputs:
                n_max_seq_inputs = len(s)
        for s_idx, s in enumerate(seq_sleepinputs):        # 新增睡眠分期结果
            if len(s) > n_max_seq_inputs:
                n_max_seq_inputs = len(s)

        n_batch_seqs = int(math.ceil(n_max_seq_inputs / seq_length))

        # For each batch sequence (size=seq_length)
        for b in range(n_batch_seqs):
            start_loop = True if b == 0 else False
            start_idx = b*seq_length
            end_idx = (b+1)*seq_length
            batch_eeginputs = np.zeros((batch_size, seq_length) + eeginput_sample_shape, dtype=np.float32)
            batch_eoginputs = np.zeros((batch_size, seq_length) + eoginput_sample_shape, dtype=np.float32)
            batch_sleepinputs = np.zeros((batch_size, seq_length) + sleepinput_sample_shape, dtype=np.int)   # 新增睡眠分期结果
            batch_targets = np.zeros((batch_size, seq_length) + target_sample_shape, dtype=np.int)
            batch_weights = np.zeros((batch_size, seq_length), dtype=np.float32)
            batch_seq_len = np.zeros(batch_size, dtype=np.int)
            # For each subject
            for s_idx, s in enumerate(zip(seq_eeginputs, seq_eoginputs, seq_targets, seq_sleepinputs)):  # 新增睡眠分期结果
                # (seq_len, sample_shape)
                each_seq_eeginputs = s[0][start_idx:end_idx]
                each_seq_eoginputs = s[1][start_idx:end_idx]
                each_seq_targets = s[2][start_idx:end_idx]
                each_seq_sleepinputs = s[3][start_idx:end_idx]  # 新增睡眠分期结果

                batch_eeginputs[s_idx, :len(each_seq_eeginputs)] = each_seq_eeginputs
                batch_eoginputs[s_idx, :len(each_seq_eoginputs)] = each_seq_eoginputs
                batch_sleepinputs[s_idx, :len(each_seq_sleepinputs)] = each_seq_sleepinputs  # 新增睡眠分期结果
                batch_targets[s_idx, :len(each_seq_targets)] = each_seq_targets
                batch_weights[s_idx, :len(each_seq_eeginputs)] = 1
                batch_seq_len[s_idx] = len(each_seq_eeginputs)
            batch_x = batch_eeginputs.reshape((-1,) + eeginput_sample_shape)
            batch_y = batch_eoginputs.reshape((-1,) + eoginput_sample_shape)
            batch_z = batch_targets.reshape((-1,) + target_sample_shape)
            batch_s = batch_sleepinputs.reshape((-1,) + sleepinput_sample_shape)   # 新增睡眠分期结果
            batch_weights = batch_weights.reshape(-1)
            # print('batch_x',batch_x.shape)
            # print('batch_y', batch_x.shape)
            # print('batch_z', batch_x.shape)
            # print('batch_s', batch_x.shape)
            yield batch_x, batch_y, batch_z, batch_s, batch_weights, batch_seq_len, start_loop      # 新增睡眠分期结果
            # yield batch_x, batch_y, batch_z, batch_weights, batch_seq_len, start_loop
