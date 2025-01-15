import music21 as m21
from music21 import stream
import numpy as np
from matplotlib import pyplot as plt
import copy

def binarise_output(output):
    # output is a set of scores: [batch size , steps , pitches , tracks]
    max_pitches = np.argmax(output, axis=3)
    return max_pitches


def clean_up_voices(score):
    if len(score.parts) < 4:
        print("Score does not have 4 voices")
        return score

    detected_key = score.analyze('key')

    clefs = [
        m21.clef.TrebleClef(),
        m21.clef.TrebleClef(),
        m21.clef.BassClef(),
        m21.clef.BassClef()
    ]

    ranges = [
        (m21.pitch.Pitch("C4"), m21.pitch.Pitch("A5")),  # Soprano
        (m21.pitch.Pitch("G3"), m21.pitch.Pitch("D5")),  # Alto
        (m21.pitch.Pitch("C3"), m21.pitch.Pitch("G4")),  # Tenor
        (m21.pitch.Pitch("E2"), m21.pitch.Pitch("C4")),  # Bass
    ]

    for i, part in enumerate(score.parts):
        if i < len(clefs):
            if part.measure(1) is None:
                continue
            if len(part.measure(1)) == 0:
                continue

        part.measure(1).clef = clefs[i]

        if detected_key:
            key_signature = m21.key.Key(detected_key.tonic, detected_key.mode)
            part.measure(1).insert(0, key_signature)

        low, high = ranges[i]  # Get range for voice
        for note in part.flatten().notes:
            while note.pitch < low:
                note.octave += 1  # Shift up
            while note.pitch > high:
                note.octave -= 1  # Shift down


    return score

def double_note_durations(music_score):
    """
    Doubles the duration of all notes and rests in a four-voice music21 score,
    ensuring that each part retains its structure while effectively doubling the number of bars.
    """
    new_score = stream.Score()

    for part in music_score.parts:  # Iterate over each voice/part
        new_part = stream.Part()
        for measure in part.getElementsByClass('Measure'):  # Process each measure
            new_measure = stream.Measure()
            new_measure.number = measure.number  # Retain measure numbering

            for element in measure.notesAndRests:
                new_element =  copy.deepcopy(element) # Corrected: Use deepcopy instead of clone
                new_element.duration.quarterLength *= 2  # Double duration
                new_measure.append(new_element)

            new_part.append(new_measure)  # Add the transformed measure to the new part

        new_score.append(new_part)  # Add the transformed part to the new score

    return new_score


def notes_to_midi(output,
                  n_bars, n_tracks, n_steps_per_bar,
                  filename,
                  show_types=[],
                  cleanup=clean_up_voices,
                  duration=double_note_durations
                  ):
    # Convert from raw 'output' to discrete pitches

    max_pitches = binarise_output(output)
    batch_size = len(output)
    generated_parts = []

    for score_num in range(batch_size):
        # Reshape from [n_bars * steps_per_bar, n_tracks]
        midi_note_score = max_pitches[score_num].reshape(
            [n_bars * n_steps_per_bar, n_tracks]
        )
        score_stream = m21.stream.Score()
        score_stream.append(m21.tempo.MetronomeMark(number=66))

        for i in range(n_tracks):
            track_notes = midi_note_score[:, i]
            last_pitch = int(track_notes[0])
            part_stream = m21.stream.Part()
            dur = 0.0

            for idx, pitch_val in enumerate(track_notes):
                pitch_val = int(pitch_val)
                # If pitch changes or we are at a four note boundary
                if (pitch_val != last_pitch or idx % 4 == 0) and idx > 0:
                    n = m21.note.Note(last_pitch)
                    n.duration = m21.duration.Duration(dur)
                    part_stream.append(n)
                    dur = 0.0
                last_pitch = pitch_val
                dur += 0.25

            # Add final note
            n = m21.note.Note(last_pitch)
            n.duration = m21.duration.Duration(dur)
            part_stream.append(n)
            score_stream.append(part_stream)

        # Optional Cleanup
        if cleanup is not None:
            score_stream = cleanup(score_stream)

        if duration is not None:
            score_stream = cleanup(score_stream)

        midi_filename = f"/content/drive/My Drive/music/museGAN/outputs/{filename}_{score_num}.mid"
        score_stream.write("midi", fp=midi_filename)

        for show_type in show_types:
            if show_type == 'score':
                score_stream.show()
            else:
                score_stream.show(show_type)

        # Append final score to the list
    generated_parts.append(score_stream)

    return generated_parts


def draw_bar(data, score_num, bar, part):
    plt.imshow(
        data[score_num, bar, :, :, part].transpose([1, 0]),
        origin="lower",
        cmap="Greys",
        vmin=-1,
        vmax=1,
    )
    plt.title(f'Score {score_num}, Bar {bar}, Part {part}')
    plt.show()


def draw_score(data, score_num):
    """
    Draw grid of subplots for each (bar, track) combination from data array
    data: tf/np.shape(batch, n_bars, n_steps_per_bar, n_pitches, n_tracks)
    """

    n_bars = data.shape[1]
    n_tracks = data.shape[-1]

    fig, axes = plt.subplots(
        nrows=n_tracks, n_cols=n_bars,
        figsize=(3 * n_bars, 3 * n_tracks),
        sharex=True, sharey=True
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for bar in range(n_bars):
        for track in range(n_tracks):
            ax = axes[track, bar] if n_tracks > 1 else axes[bar]
            ax.im_show(
                data[score_num, bar, :, :, track].transpose([1, 0]),
                origin='lower',
                cmap='Greys',
                vmin=-1,
                vmax=1,
            )
            ax.set_title(f'Bar {bar}, Track {track}')
    plt.tight_layout()
    plt.show()
