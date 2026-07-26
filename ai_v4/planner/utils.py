from copy import deepcopy


def deep_merge(destination: dict, source: dict) -> dict:

    if destination is None:
        destination = {}

    destination = deepcopy(destination)

    for key, value in source.items():

        if (
            key in destination
            and isinstance(destination[key], dict)
            and isinstance(value, dict)
        ):
            destination[key] = deep_merge(
                destination[key],
                value,
            )

        elif (
            key in destination
            and isinstance(destination[key], list)
            and isinstance(value, list)
        ):
            merged = destination[key][:]

            for item in value:
                if item not in merged:
                    merged.append(item)

            destination[key] = merged

        else:
            destination[key] = value

    return destination