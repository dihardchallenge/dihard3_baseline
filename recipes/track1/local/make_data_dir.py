#!/usr/bin/env python
"""Populate Kaldi data directory for diarization.

The resulting data directory contains the following files:

- wav.scp  --  Kaldi WAV script file with one entry per FLAC file located
  under ``data/flac/``
- segments  --  Kaldi segments file containing  reference speech segmentation;
  generated from the contents of the ``.lab`` files under ``data/sad/`` with
  one utterance per reference speech segment
- utt2spk  --  Kaldi ``utt2spk`` file mapping utterances to its "speaker";
  here, each recording is considered a unique speaker
- reco2num_spk  --  mapping from recordings to the oracle number of speakers
  present

Within these files, recording ids, utterance ids, and speaker ids are defined
as follows:

- recording id  --  set to the basename of the corresponding FLAC file minus
  the extension; e.g., the recording id of ``data/flac/DH_DEV_0001.flac`` is
  ``DH_DEV_0001``
- speaker id  --  the speaker id of each utterance/segment is set to the
  recording id of the recording the utterance/segment is from; that is, every
  segment from the recording ``DH_DEV_0001`` is assigned the speaker id
  ``DH_DEV_0001``
- utterance id  --  utterance ids are name according to the convention

      <recording-id>_<index>

  where  ``recording-id`` is the recording id of the segment's parent recording
  and ``index`` is the position of the segment within that recording; e.g.,
  the following are valid utterance ids:

  - DH_DEV_0001_0001
  - DH_DEV_0001_0002
  - ...
"""
import argparse
from dataclasses import dataclass
import itertools
from pathlib import Path
import sys


@dataclass
class Segment:
    """Speech segment, which may contain speech from multiple speakers.

    Parameters
    ----------
    utterance_id : str
        Unique identifier for segment.

    onset : float
        Onset in seconds of segment.

    offset : float
        Offset in seconds of segment.

    label : str
        Segment label; e.g., "speech".
    """
    utterance_id: str
    onset: float
    offset: float
    label: str

    @property
    def duration(self):
        """Duration of segment in seconds."""
        return self.offset - self.onset


@dataclass
class RTTMTurn:
    """Speaker turn from RTTM file.

    Parameters
    ----------
    recording_id : str
        Recording turn is from.

    speaker_id : str
        Speaker of turn.

    onset : float
        Onset of turn in seconds.

    offset : float
        Offset of turn in seconds.

    rttm_line : str
        Original line of turn in source RTTM file.
    """
    recording_id: str
    speaker_id: str
    onset: float
    offset: float
    rttm_line: str


@dataclass
class Recording:
    """
    Parameters
    ----------
    recording_id : str
        Recording id.

    lab_path : Path
        Path to label file containing speech segmentation.

    audio_path : Path
        Path to audio file.

    rttm_path : Path, optional
        Path to RTTM file containing diarization.
        (Default: None)
    """
    recording_id: str
    lab_path: Path
    audio_path: Path
    rttm_path: Path=None

    def __post_init__(self):
        self._segments = None
        self._turns = None

    @property
    def segments(self):
        """Speech segments."""
        if self._segments is None:
            self._segments = read_label_file(self.lab_path)
        return self._segments

    @property
    def turns(self):
        """Speaker turns in recording's diarization."""
        if self._turns is None:
            if self.rttm_path is None or not self.rttm_path.exists():
                raise AttributeError(
                    f'diarization not available for recording '
                    f'"{self.recording_id}"')
            self._turns = read_rttm_file(self.rttm_path)
        return self._turns

    @property
    def speakers(self):
        """Speakers present on recording."""
        return set(turn.speaker_id for turn in self.turns)

    @property
    def num_speakers(self):
        """Number of speakers present on recording."""
        return len(self.speakers)


    @staticmethod
    def load_recordings(sad_dir, flac_dir, rttm_dir=None):
        """Load recordings from FLAC, SAD, and RTTM files.

        Parameters
        ----------
        sad_dir : str
            Path to directory containing SAD label files.

        audio_dir : str
            Path to directory containing FLAC files.

        rttm_dir : str, optional
            Path to directory containing RTTM files.
            (Default: None)

        Returns
        -------
        list of Recording
            Recordings.
        """
        flac_dir = Path(flac_dir)
        recordings = []
        flac_paths = sorted(flac_dir.glob('*.flac'))
        for flac_path in flac_paths:
            recording_id = flac_path.stem
            lab_path = Path(sad_dir, recording_id + '.lab')
            rttm_path = None
            if rttm_dir is not None and rttm_dir.exists():
                rttm_path = Path(rttm_dir, recording_id + '.rttm')
            recordings.append(Recording(
                recording_id, lab_path, flac_path, rttm_path))
        return recordings


def read_label_file(lab_path):
    """Load segments from label file.

    Parameters
    ----------
    lab_path : Path
        Path to label file containing segments.

    Returns
    -------
    list of Segment
        Segments.
    """
    lab_path = Path(lab_path)
    recording_id = lab_path.stem
    with open(lab_path, 'r') as f:
        segs = []
        for n, line in enumerate(f, start=1):
            onset, offset, label = line.strip().split()
            utterance_id = f'{recording_id}_{n:04d}'
            segs.append(
                Segment(utterance_id, onset, offset, label))
    return segs


def read_rttm_file(rttm_path):
    """Load speaker turns from RTTM file.

    Parameters
    ----------
    rttm_path : Path
       Path to RTTM file.

    Returns
    -------
    list of Turn
       Speaker turns.
    """
    with open(rttm_path, 'r') as f:
        turns = []
        for line in f:
            fields = line.strip().split()
            recording_id = fields[1]
            onset = float(fields[3])
            offset = onset + float(fields[4])
            speaker_id = fields[7]
            turns.append(
                RTTMTurn(recording_id, speaker_id, onset, offset, line))
    return turns


def write_rttm_file(rttm_path, turns):
    """Write speaker turns to RTTM file.

    Parameters
    ----------
    rttm_path : Path
        Path to output RTTM file.

    turns : list of Turn
        Speaker turns.
    """
    with open(rttm_path, 'w') as f:
        turns = sorted(
            turns, key=lambda x: (x.recording_id, x.onset, x.offset))
        for turn in turns:
            f.write(turn.rttm_line)


def write_wav_script_file(wav_scp_path, recordings, target_sr=16000):
    """Write Kaldi ``wav.scp`` file.

    Parameters
    ----------
    wav_scp_path : Path
        Path to output script file.

    recordings : list of Recording
        Recordings.

    target_sr : int, optional
        Resample audio to ``target_sr`` Hz.
        (Default: 16000)
    """
    with open(wav_scp_path, 'w') as f:
        for recording in recordings:
            efname = (f'sox {recording.audio_path} -t wav -b 16 - '
                      f'rate {target_sr} remix 1 |')
            line = f'{recording.recording_id} {efname}\n'
            f.write(line)


def write_utt2spk(utt2spk_path, recordings):
    """Write ``utt2spk`` file.

    Each utterance is assigned its corresponding recording id as the speaker.
    """
    with open(utt2spk_path, 'w') as f:
        recordings = sorted(recordings, key=lambda x: x.recording_id)
        for recording in recordings:
            segments = sorted(recording.segments, key=lambda x: x.utterance_id)
            for segment in segments:
                line = f'{segment.utterance_id} {recording.recording_id}\n'
                f.write(line)


def write_segments_file(segments_path, recordings):
    """Write ``segments`` file."""
    with open(segments_path, 'w') as f:
        recordings = sorted(recordings, key=lambda x: x.recording_id)
        for recording in recordings:
            segments = sorted(recording.segments, key=lambda x: x.utterance_id)
            for segment in segments:
                line = (f'{segment.utterance_id} {recording.recording_id} '
                        f'{segment.onset} {segment.offset}\n')
                f.write(line)


def write_reco2num_spk(reco2num_spk_path, recordings):
    """Write ``reco2num_spk`` file."""
    with open(reco2num_spk_path, 'w') as f:
        recordings = sorted(recordings, key=lambda x: x.recording_id)
        for recording in recordings:
            line = f'{recording.recording_id} {recording.num_speakers}\n'
            f.write(line)


def warning(msg):
    """Print WARNING to STDERR."""
    print(f'WARNING: {msg}', file=sys.stderr)

    
def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description='Prepare data directory for KALDI experiments.',
        add_help=True)
    parser.add_argument(
        'data_dir', metavar='data-dir', type=Path,
        help='path to output data directory')
    parser.add_argument(
        'flac_dir', metavar='flac-dir', type=Path,
        help='path to source FLAC directory')
    parser.add_argument(
        'sad_dir', metavar='sad-dir', type=Path,
        help='path to source SAD directory')
    parser.add_argument(
        '--rttm-dir', metavar='PATH', type=Path, nargs=None, default=None,
        help='path to source RTTM directory')
    parser.add_argument(
        '--target-sr', metavar='SR', type=int, default=16000,
        help='resample audio to SR Hz (default: %(default)s)')
    parser.add_argument(
        '--rec-ids', metavar='FILE', type=Path, default=None,
        help='read recordings to keep from FILE; ignore all other recordings')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    has_rttm = args.rttm_dir is not None and args.rttm_dir.exists()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    recordings = Recording.load_recordings(
        args.sad_dir, args.flac_dir, args.rttm_dir)
    if args.rec_ids is not None:
        with open(args.rec_ids, 'r') as f:
            keep_rec_ids = {line.strip() for line in f}
        recordings = [rec for rec in recordings
                      if rec.recording_id in keep_rec_ids]
    write_wav_script_file(
        Path(args.data_dir, 'wav.scp'), recordings, target_sr=args.target_sr)
    write_segments_file(
        Path(args.data_dir, 'segments'), recordings)
    write_utt2spk(
        Path(args.data_dir, 'utt2spk'), recordings)
    if has_rttm:
        write_reco2num_spk(
            Path(args.data_dir, 'reco2num_spk'), recordings)
    if has_rttm:
        turns = list(itertools.chain.from_iterable(
            recording.turns for recording in recordings))
        write_rttm_file(
            Path(args.data_dir, 'rttm'), turns)


if __name__ == '__main__':
    main()
