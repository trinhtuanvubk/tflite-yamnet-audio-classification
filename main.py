import scenario


def main():
    args = scenario.get_args()
    method = getattr(scenario, args.scenario)
    method(args)


if __name__ == "__main__":
    main()
