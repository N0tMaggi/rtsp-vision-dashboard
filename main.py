from securitycam import config
from securitycam.webapp import app


def main() -> None:
    app.run(host=config.APP_HOST, port=config.APP_PORT, debug=config.APP_DEBUG)


if __name__ == "__main__":
    main()
