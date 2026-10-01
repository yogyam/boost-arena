# Security

Boost Arena runs model files sent in by strangers and holds a private key that opens them. If you find a way to break either of those promises, please tell us privately first.

## Reporting

Use GitHub's private vulnerability reporting on this repository ("Security" tab, "Report a vulnerability"), or email yogyamehrotra@gmail.com. Please don't open a public issue for anything that could be used against the service before it is fixed. We aim to acknowledge a report within a week; this is a volunteer project.

Things we would like to hear about:

- A model file that passes `boost-arena check` but crashes, stalls or exhausts the scoring process, or reads anything it should not.
- A way to publish a score the scoring service did not produce, or to publish one under another entry's name.
- A way to open a sealed model without the private key, or to get the key out of the scoring job.
- Anything on the website that runs script from another origin or shows content an entrant did not write.

## What is in place

- Submitted models are ONNX files restricted to an allow-list of feed-forward operations, with limits on file size, parameters, intermediate values and decision time, checked before onnxruntime loads the file.
- The private key is given to one workflow step, which opens the sealed models into a folder outside the checkout and removes the key from its environment. The steps that run the models have no key, and run each model in its own process with a time and memory limit.
- Everything the scoring job produces is checked against the repository by a separate job before it is published: a result must be for the manifest on `main`, its numbers must agree with each other, and a duel must be between two bots with current scores.
- A sealed file is bound to the slug and GitHub login it was sealed for.
- The website loads no script from other hosts and sets a content security policy; every entrant-written string is escaped.
- Actions are pinned to commits, and Dependabot proposes updates.

## If the private key is ever exposed

The maintainer runs `boost-arena keygen`, replaces `PUBLIC_KEY` and the `BOOST_ARENA_PRIVATE_KEY` secret, and announces that every entrant must seal their model again with the new key. Existing scores stay; a bot that is not re-sealed sits out future duels and re-scorings.
