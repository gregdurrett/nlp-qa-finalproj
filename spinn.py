import torch
from torch import nn

class SPINN(nn.module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.reduce = Reduce(config.d_hidden, config.d_tracker)
    if config.d_tracker is not None:
      self.tracker = Tracker(config.d_hidden, config.d_tracker)

  def forward(self, buffers, transitions):
    buffers = [list(torch.split(b.squeeze(1), 1, 0)) for b in torch.split(buffers, 1, 1)]
    stacks = [[buf[0], buf[0]] for buf in buffers]
    if hasattr(self, 'tracker'):
      self.tracker.reset_state()
    for trans_batch in transitions:
      if hasattr(self, 'tracker'):
        tracker_states, _ = self.tracker(buffers, stacks)
      else:
        tracker_states = itertools.repeat(None)
        lefts, rights, trackings = [], [], []
        batch = zip(trans_batch, buffers, stacks, tracker_states)
        for transition, buf, stack, tracking in batch:
            if transition == SHIFT:
                stack.append(buf.pop())
            elif transition == REDUCE:
                rights.append(stack.pop())
                lefts.append(stack.pop())
                trackings.append(tracking)

        if rights:
            reduced = iter(self.reduce(lefts, rights, trackings))
            for transition, stack in zip(trans_batch, stacks):
                if transition == REDUCE:
                    stack.append(next(reduced))
    return [stack.pop() for stack in stacks]
