=========
Changelog
=========

Version 0.9.0
===========

- Finished core features:
	- Attention Mechanism
	- Diffusion Graph Convolution
	- GRU based on the above
	- reimplementation of the model used in https://milets19.github.io/papers/milets19_paper_8.pdf

Version 0.9.1
===========

- Wrote README.md
- Added requirements
- uploading to TestPyPi

Version 0.9.2
===========

- fixing README.md
- removed pyscaffold spam from setup.cfg
- uploading to TestPyPi and testing if it installs correctly

Version 0.9.3
===========

- Final upload to real pypi and final test

Version 1.0.0
===========

- All good. Ready for release.

Version 1.0.1
===========

- fixing naming bug where we could not use more than one AttentionMechanism due to
  all such mechanisms having the same name but names need to be unique.

Version 1.0.2
===========

- Added RaggedTensorFeeder utensil that is able to feed a ragged tensor as
data to an RNN model that can only train on non ragged batches.

Version 1.0.3
===========

- Fixed using an average layer for only one attention head.
- Fixed issue with many weights not being trained due to kwargs being passed to the parent object in the layer definitions.
