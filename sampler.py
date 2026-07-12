"""Distributed graph sampler wrapping DGL's DistNodeDataLoader."""

import dgl.dataloading


class DistSampler:
    def __init__(self, g, train_nid, fan_out, batch_size,
                 num_workers=0, prefetch_factor=2, generator=None):
        self.train_nid = train_nid
        sampler = dgl.dataloading.NeighborSampler(
            [int(x) for x in fan_out.split(",")])

        kwargs = {"batch_size": batch_size, "shuffle": True, "drop_last": False}
        if num_workers and num_workers > 0:
            kwargs["num_workers"] = num_workers
            if prefetch_factor and prefetch_factor > 0:
                kwargs["prefetch_factor"] = prefetch_factor

        # Seeded shuffling (spec I6). DistNodeDataLoader in some DGL versions
        # does not accept `generator`; global torch/numpy seeding from
        # helpers.set_all_seeds covers those (its shuffle draws from the
        # globally seeded RNG), so a TypeError here is non-fatal.
        if generator is not None:
            try:
                self.dataloader = dgl.dataloading.DistNodeDataLoader(
                    g, train_nid, sampler, generator=generator, **kwargs)
                return
            except TypeError:
                pass
        self.dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, sampler, **kwargs)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        bs = self.dataloader.batch_size
        return (self.train_nid.shape[0] + bs - 1) // bs
