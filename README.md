Check Labels Action
===================

This action checks the labels of a Pull Request and will succeed or fail based
on the configuration.

Inputs
------

### `access_token`
An optional [personal access token], required for private repositories and to
decrease rate limiting.

[personal access token]: https://github.com/settings/tokens

### `set_labels`
Comma-separated list of labels required to be set. Optional. Globing syntax is
possible for the label name, as defined in [fnmatch].

### `unset_labels`
Comma-separated list of labels required not to be set. Optional. Globing syntax
is possible for the label name, as defined in [fnmatch].

### `cond_labels`
Comma-separated list of (label,condition) tuples for labels introducing a
condition. Optional. Globing syntax is possible for the label name, as defined
in [fnmatch].

### `missing_approvals_label`
Name of a label that this action will set/unset according to the state of
required approvals. The label will be set if approvals are missing, and unset
if there are sufficient approvals.

#### Supported conditions
- `review.approvals>x`: If the label is set in the Pull Request it requires more
  than `x` approving reviews for the action to succeed

[fnmatch]: https://docs.python.org/3/library/fnmatch.html

# Examples

We recommend the following workflow triggers:

```yml
on:
  pull_request:
    types: [opened, reopened, labeled, unlabeled]
  pull_request_review:
    types: [submitted, dismissed]
```

The action will fail if "REQUIRE" and "MANDATORY" are not set, if any label
starting with "INVALID" is set, or if "NEEDS >1 ACK" is set, but the PR only has
one or no approval:

```yml
uses: RIOT-OS/check-labels-action@v1.0.0
with:
    access_token: '${{ secrets.GITHUB_ACCESS_TOKEN }}'
    set_labels: 'REQUIRE, MANDATORY'
    unset_labels: 'INVALID*'
    cond_labels: '(NEEDS >1 ACK,review.approvals > 1)'
```
