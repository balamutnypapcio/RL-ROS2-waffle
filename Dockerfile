# Używamy oficjalnego obrazu Desktop (z GUI)
FROM osrf/ros:humble-desktop

# Ustawienie powłoki na bash
SHELL ["/bin/bash", "-c"]

# Instalacja dodatkowych przydatnych narzędzi + COLCON EXTENSIONS
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    nano \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tworzymy folder workspace'u wewnątrz kontenera
WORKDIR /ros2_ws

# Automatyczne dodanie source setup.bash do każdego nowego terminala
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Komenda domyślna
CMD ["bash"]
