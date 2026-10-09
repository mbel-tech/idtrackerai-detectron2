"""Identity swaps on blobs whose user_generated_identities are out of step."""

import pytest

from idtrackerai import Blob


def make_blob(frame, identities, fragment=0):
    blob = Blob.__new__(Blob)
    blob.identities_corrected_closing_gaps = list(identities)
    blob.interpolated_centroids = [(10.0 * i, 0.0) for i in range(len(identities))]
    blob.user_generated_identities = None
    blob.user_generated_centroids = None
    blob.frame_number = frame
    blob.fragment_identifier = fragment
    blob.next = ()
    blob.previous = ()
    return blob


def test_swap_with_short_user_list():
    blob = make_blob(0, [1, 2])
    blob.user_generated_identities = []  # falsy, shorter than the identities
    blob.user_generated_centroids = [None, None]
    blob.swap_identity(1, 2, (0.0, 0.0))
    assert blob.all_final_identities == [2, 1]


def test_swap_with_itself_is_noop():
    blob = make_blob(0, [1, 2])
    blob.swap_identity(1, 1, (0.0, 0.0))
    assert blob.all_final_identities == [1, 2]


def test_propagation_swaps_whole_fragment():
    blobs = [make_blob(f, [1, 2]) for f in range(3)]
    for a, b in zip(blobs, blobs[1:]):
        a.next, b.previous = (b,), (a,)
    first, last = blobs[1].propagate_swap_identity(1, 2, (0.0, 0.0))
    assert (first, last) == (0, 2)
    assert all(b.all_final_identities == [2, 1] for b in blobs)


def test_propagation_rolls_back_on_failure(monkeypatch):
    blobs = [make_blob(f, [1, 2]) for f in range(3)]
    for a, b in zip(blobs, blobs[1:]):
        a.next, b.previous = (b,), (a,)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(blobs[2], "swap_identity", boom)
    with pytest.raises(RuntimeError):
        blobs[0].propagate_swap_identity(1, 2, (0.0, 0.0))
    assert all(b.all_final_identities == [1, 2] for b in blobs)
