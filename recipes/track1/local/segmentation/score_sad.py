#!/usr/bin/env python
"""Score SAD system output.

To evaluate system output against reference speech segmentation, run:

    python score.py ref_segments sys_segments recordings_table

where:

- ref_segments  --  path to Kaldi segments file containing the reference
  speech segments
- sys_segments  --  path to Kaldi segments file containing the system
  speech segments
- recordings_table  --   path to recordings metadata file; **MUST**
  contain the columns ``uri`` and `` domain``

The script will then pair up reference/system recordings with the same
unique resource identifier (URI) and compute the following metrics overall
and on a per-domain basis:

- detection cost function (DCF) [1]
- detection error rate (DER) [2]
- accuracy
- false alarm rate
- miss rate

These metrics will be output to STDOUT as a table with the following 10
columns:

- domain  --  recording domain; overall results are reported under the domain
  ``OVERALL``
- dcf  --  detection cost function in percent
- der  --  detection error rate in percent
- accuracy  --  accuracy in percent
- fa  --  false alarm rate in percent
- miss  --  miss rate in percent
- fa duration  --  the total duration in seconds of all false alarms
- miss duration  --  the total duration in seconds of all misses
- speech duration  --  the total duration in seconds of speech according to
  the reference segmentation
- nonspeech duration  --  the total duration in seconds of non-speech
  according to the reference segmentation

For instance:

    domain                 dcf    der    accuracy     fa    miss    fa duration    miss duration    speech duration    nonspeech duration
    -------------------  -----  -----  ----------  -----  ------  -------------  ---------------  -----------------  --------------------
    audiobooks            5.89   6.93       94.62  19.80    1.25         323.54            71.38            5702.56               1634.39
    broadcast_interview   9.78  11.80       90.86  29.01    3.37         476.68           190.28            5653.88               1643.22
    child                21.82  47.59       72.95  43.42   14.62        1255.22           556.48            3806.75               2890.90
    clinical             14.84  27.09       83.77  21.79   12.52         654.79           562.74            4494.83               3005.36
    court                 9.13   8.10       93.26  31.00    1.83         382.67           112.09            6109.34               1234.45
    maptask               5.86  10.22       93.36  11.64    3.94         302.37           189.44            4810.31               2598.28
    meeting              52.45  69.82       42.54   0.98   69.61          11.66          3851.39            5532.64               1190.27
    restaurant           40.48  50.07       55.85  19.58   47.44         171.35          3095.56            6525.19                874.91
    socio_field          14.24  17.28       86.60  38.86    6.04         710.90           381.62            6322.59               1829.45
    socio_lab            33.14  43.23       67.13  52.94   26.55         927.09          1475.27            5557.63               1751.27
    webvideo             24.80  33.95       74.98  37.13   20.69         724.54          1130.55            5464.26               1951.33
    OVERALL              21.73  29.27       78.21  28.83   19.37        5940.80         11616.82           59979.98              20603.84

By default the scoring regions for each recording will be determined
automatically from the from the reference and speaker segmentations. However,
it is possible to specify explicit scoring regions using a NIST un-partitioned
evaluation map (UEM) file and the ``-u`` flag. For instance:

    python score.py -u all.uem ref_segments sys_segments recordings_table

The scoring regions may also be modified by use the of the ``--collar`` flag,
which controls whether or not a NIST-style scoring collar is applied prior to
computing the metrics. By default, no collar is used, but if a duration ``DUR``
is passed, no scoring will be performed withing ``+-DUR`` seconds of reference
segmentation turn boundaries; for instance:

    python score.py -u all.uem --collar 0.250 ref_segments sys_segments recordings_table

will eliminate from scoring all regions within 250 ms of a reference turn
boundary.

Some basic control of the formatting of the results table is possible via the
``--precision`` and ``--table_format`` flags. The former controls the number
of decimal places printed for floating point numbers, while the latter
controls the table format. For a list of valid table formats plus example
outputs, consult the documentation for the ``tabulate`` package (specifically,
the documentation of the ``tablemt`` argument):

    https://pypi.python.org/pypi/tabulate

References
----------
[1] "OpenSAT19 Evaluation Plan v2." https://www.nist.gov/system/files/documents/2018/11/05/opensat19_evaluation_plan_v2_11-5-18.pdf
[2] Bredin, H. (2017). "pyannote.metrics: A Toolkit for Reproducible Evaluation,
  Diagnostic, and Error Analysis of Speaker Diarization Systems." In Proc. of
  INTERSPEECH 2017. pp. 3587-3591.
"""
import argparse
from dataclasses import dataclass
from functools import partial
import multiprocessing as mp
from pathlib import Path
import sys

import pandas as pd
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.detection import (
    DetectionCostFunction, DetectionErrorRate, DCF_NAME, DCF_POS_TOTAL,
    DCF_NEG_TOTAL, DCF_FALSE_ALARM, DCF_MISS)
from tabulate import tabulate



@dataclass
class Recording:
    """Recording.

    Parameters
    ----------
    uri : str
        Unique resource identifier (URI) for recording.

    ref_speech : Annotation
        Reference speech/non-speech annotation.

    sys_speech : Annotation
        System speech/non-speech annnotation.

    annnotated : Timeline
        Annotated regions of file; scoring will be limited to these regions.
    """
    uri: str
    ref_speech: Annotation
    sys_speech: Annotation
    annotated: Timeline

    @staticmethod
    def annotations_to_recordings(ref_annotations, sys_annotations,
                                  annotated=None, uris=None):
        """Extract ``Recording`` instances from paired annotations.

        Parameters
        ----------
        ref_annotations : dict
            ``ref_annotations[uri]`` is the reference speech annotation for
            recording ``uri``.

        sys_annotations : dict
            ``sys_annotations[uri]`` is the system speech annotation for
         recording ``uri``.

        annotated : dict, optional
            ``annotated[uri]`` is the timeline of scoring regions for
            recording``uri``; if ``annotated`` is ``None``, then the scoring
            regions will be approximated as the smallest extent containing all
            reference/system segments.

        uris : iterable of str, optional
            URIs of recordings to score. If ``None``, determined
            automatically from ``ref_annotations``.

        Returns
        -------
        list of Recording
            Recordings.
        """
        annotated = {} if annotated is None else annotated

        # Determine recordings to score.
        if uris is None:
            uris = ref_annotations.keys()
        uris = set(uris)

        # Check for missing recordings.
        for uri in uris:
            # Only check for presence in reference as we know speech is always
            # present in those segmentations, whereas system output could
            # concievably not output speech for some recordings, resulting in
            # no lines in the segments file.
            if uri not in ref_annotations:
                raise(ValueError(
                    f'"ref_annotations" missing Recording "{uri}".'))

        # Group.
        recordings = []
        for uri in sorted(uris):
            ref_ann = ref_annotations[uri]
            sys_ann = Annotation(uri=uri)
            if uri in sys_annotations:
                sys_ann = sys_annotations[uri]
            annotated_t = annotated.get(uri, None)
            if annotated_t is None:
                # Approximate scoring regions from smallest extent containing
                # all reference/system segments.
                ref_extent = ref_ann.get_timeline(copy=False).extent()
                sys_extent = sys_ann.get_timeline(copy=False).extent()
                annotated_t = ref_extent | sys_extent
            recordings.append(Recording(
                uri, ref_ann, sys_ann, annotated_t))

        return recordings


def read_segments_file(segments_path):
    """Read speech/non-speech annotations from Kaldi segments file."""
    columns = ['utterance_uri', 'recording_uri', 'onset', 'offset']
    segs_df = pd.read_csv(
        segments_path, header=None, sep=' ', names=columns)
    annotations = {}
    for recording_uri, segs in segs_df.groupby('recording_uri'):
        records = [(Segment(seg.onset, seg.offset), '_', 'speech')
                   for seg in segs.itertuples(index=False)]
        ann = Annotation.from_records(records, uri=recording_uri)
        annotations[recording_uri] = ann

    return annotations


def read_uem_file(uem_path):
    """Read scoring regions from unpartitioned evaluation map (UEM) file."""
    columns = ['uri', 'channel', 'onset', 'offset']
    df = pd.read_csv(uem_path, header=None, sep=' ', names=columns)
    annotated = {}
    for uri, segs in df.groupby('uri'):
        annotated[uri] = Timeline(
            Segment(seg.onset, seg.offset) for seg in segs.itertuples(False))
    return annotated


def load_domains(table_path):
    """Load domains from recordings table."""
    df = pd.read_csv(table_path, header=0, sep='\t')
    domains = {}
    for dname, recordings in df.groupby('domain').uri:
        domains[dname] = set(recordings)
    return domains


def sum_metrics(*metrics):
    """Sum an iterable of metrics."""
    cls = metrics[0].__class__
    result = cls()
    for metric in metrics:
        if metric.__class__ != cls:
            raise ValueError('All metrics must be of same type.')
        result.results_.extend(metric.results_)
        result.uris_.update(metric.uris_)
        for cname in metric.components_:
            result.accumulated_[cname] += metric.accumulated_[cname]
    return result


def _score_one_recording(recording, metrics):
    for mname in sorted(metrics):
        metric = metrics[mname]
        metric(
            recording.ref_speech, recording.sys_speech,
            uem=recording.annotated)
    return metrics


def score_recordings(recordings, metrics, domains=None, n_jobs=1):
    """Score recordings.

    Parameters
    ----------
    recordings : list of Recording
        Recordings to score.

    metrics : dict
        Mapping from metric names to instances.

    domains : dict, optional
        ``domains[dname]`` is the set of all URIs of recordings  belonging to
        domain ``dname``.
        (Default: None)

    n_jobs : int, optional
        Number of parallel jobs.
        (Default: 1)

    Returns
    -------
    per_domain_metrics : dict
        ``per_domain_metrics[dname]`` is the dictionary of metrics
        corresponding to domain ``dname``; that is, a mapping from metric
        names to metrics. Overall results are associated with the domain
        named "OVERALL".
    """
    domains = {} if domains is None else domains
    domains['OVERALL'] = {recording.uri for recording in recordings}

    # Accumulate raw stats for seach recording.
    results = {}  # recording URI ==> metrics for that recording.
    with mp.Pool(n_jobs) as pool:
        f = partial(_score_one_recording, metrics=metrics)
        for recording, result in zip(recordings, pool.imap(f, recordings)):
            results[recording.uri] = result

    # Aggregate by domain.
    per_domain_metrics = {}  # domain name ==> metrics for that domain.
    for dname in domains:
        domain_results = []
        for uri in sorted(domains[dname]):
            if uri not in results:
                continue
            domain_results.append(results[uri])
        domain_metrics = {}
        for metric_name in metrics:
            domain_metrics[metric_name] = sum_metrics(
                *[res[metric_name] for res in domain_results])
        per_domain_metrics[dname] = domain_metrics

    return per_domain_metrics


def get_scores_dataframe(metrics, domain_name=None):
    """Return ``DataFrame`` containing scoring results."""
    # Extract values of metrics + components for each domain.
    der = metrics['der'].report(display=False)
    dcf = metrics['dcf'].report(display=False)
    scores_df = pd.DataFrame({
        'uri' : dcf.index,
        'dcf' : dcf[DCF_NAME, '%'],
        'der' : der[metrics['der'].name, '%'],
        'speech_dur' : dcf[DCF_POS_TOTAL, ''],
        'nonspeech_dur' : dcf[DCF_NEG_TOTAL, ''],
        'fa_dur' : dcf[DCF_FALSE_ALARM, ''],
        'miss_dur' : dcf[DCF_MISS, ''],
        })

    # Drop individual recordings.
    scores_df = scores_df[scores_df.uri == 'TOTAL']

    # Rename row with overall results.
    if domain_name is not None:
        scores_df.uri.iloc[-1] = domain_name

    # Augment with miss rate, false alarm rate, and accuracy.
    scores_df['fa_rate'] = 100.*(scores_df.fa_dur/scores_df.nonspeech_dur)
    scores_df['miss_rate'] = 100.*(scores_df.miss_dur/scores_df.speech_dur)
    total_dur = scores_df.speech_dur + scores_df.nonspeech_dur
    correct_dur = total_dur - (scores_df.fa_dur + scores_df.miss_dur)
    scores_df['accuracy'] = 100.*correct_dur/total_dur
    scores_df.reset_index(drop=True, inplace=True)

    # Reorder/rename columns for prettier display.
    scores_df = scores_df[
        ['uri', 'dcf', 'der', 'accuracy', 'fa_rate', 'miss_rate',
         'fa_dur', 'miss_dur', 'speech_dur', 'nonspeech_dur']]
    columns = {
        'uri' : 'domain',
        'fa_rate' : 'fa',
        'miss_rate' : 'miss',
        'fa_dur' : 'fa duration',
        'miss_dur' : 'miss duration',
        'speech_dur' : 'speech duration',
        'nonspeech_dur' : 'nonspeech duration',
        }
    scores_df.rename(columns=columns, inplace=True)

    return scores_df



def main():
    parser = argparse.ArgumentParser(
        description='score SAD output', add_help=True)
    parser.add_argument(
        'ref_segments', metavar='ref-segments', type=Path,
        help='path to reference segments file')
    parser.add_argument(
        'sys_segments', metavar='sys-segments', type=Path,
        help='path to system segments file')
    parser.add_argument(
        'recordings_table', metavar='recordings-table', type=Path,
        help='path to recordings table')
    parser.add_argument(
        '-u,--uem', metavar='FILE', default=None, type=Path, dest='uem_path',
        help='un-partitioned evaluation map file (default: %(default)s)')
    parser.add_argument(
        '--collar', nargs=None, default=0.0, type=float, metavar='DUR',
        help='collar size in seconds (default: %(default)s)')
    parser.add_argument(
        '--precision', nargs=None, default=2, type=int, metavar='DIGITS',
        help='number of decimal places to print (default: %(default)s)')
    parser.add_argument(
        '--table-format', nargs=None, default='simple',
        metavar='FMT',
        help='tabulate table format (default: %(default)s)')
    parser.add_argument(
        '--n-jobs', nargs=None, default=1, type=int, metavar='JOBS',
        help='number of parallel jobs to run (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # Load annotations, scoring regions, etc.
    ref_annotations = read_segments_file(args.ref_segments)
    sys_annotations = read_segments_file(args.sys_segments)
    annotated = {}
    if args.uem_path:
        annotated = read_uem_file(args.uem_path)

        ref_annotations = {uri:ann for uri, ann in ref_annotations.items()
                           if uri in annotated}
        sys_annotations = {uri:ann for uri, ann	in sys_annotations.items()
                           if uri in annotated}
    recordings = Recording.annotations_to_recordings(
        ref_annotations, sys_annotations, annotated=annotated)
    domains = load_domains(args.recordings_table)

    # Score in parallel.
    collar = 2*args.collar # Accommodation for how pyannote defines collar.
    kwargs = {
        'collar' : collar,
        'skip_overlap' : False,
        'parallel' : False}
    metrics = {
        'dcf' : DetectionCostFunction(
            fa_weight=0.25, miss_weight=0.75, **kwargs),
        'der' : DetectionErrorRate(**kwargs)}
    per_domain_metrics = score_recordings(
        recordings, metrics, domains=domains, n_jobs=args.n_jobs)

    # Report metrics as table on STDOUT.
    domain_dfs = []
    for dname in per_domain_metrics:
        domain_dfs.append(get_scores_dataframe(
            per_domain_metrics[dname], domain_name=dname))
    scores_df = pd.concat(domain_dfs)
    tbl = tabulate(
        scores_df, showindex=False, headers=scores_df.columns,
        tablefmt=args.table_format, floatfmt=f'.{args.precision}f')
    print(tbl)


if __name__ == '__main__':
    main()
