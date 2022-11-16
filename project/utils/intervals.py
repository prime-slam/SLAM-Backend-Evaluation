import csv
from typing import List, Tuple


def load_evaluation_intervals(path: str, total_frames: int) -> List[Tuple[int, int]]:
    result = []
    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            from_id, to_id = row[:2]
            result.append((int(from_id), int(to_id)))

    return result


def dump_evaluation_intervals(path: str, intervals: List[Tuple[int, int]]):
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for interval in intervals:
            writer.writerow(interval)


def ids_list_to_intervals(ids: List[int]) -> List[Tuple[int, int]]:
    result = []
    interval_start = None
    prev_id = None
    for index, identifier in enumerate(ids):
        if prev_id is None or prev_id != identifier - 1:
            if interval_start is not None:
                result.append((interval_start, prev_id))
            interval_start = identifier

        prev_id = identifier

    if interval_start is not None:
        result.append((interval_start, prev_id))

    return result
