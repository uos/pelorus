# Maintainer: Christopher Sieh (stelzo) <stelzo@steado.de>
# Template file — CI replaces VERSION_PLACEHOLDER, ARM_SHA256_PLACEHOLDER,
# and X86_SHA256_PLACEHOLDER before publishing to the tap.
class Pelorus < Formula
  desc "Highly efficient Lidar Inertial Odometry"
  homepage "https://codeberg.org/stelzo/pelorus"
  version "VERSION_PLACEHOLDER"
  license any_of: ["MIT", "Apache-2.0"]

  on_macos do
    on_arm do
      url "https://uos-robotics.codeberg.page/ppa/bin/pelorus-#{version}-macos-arm64.tar.gz"
      sha256 "ARM_SHA256_PLACEHOLDER"
    end

    on_intel do
      url "https://uos-robotics.codeberg.page/ppa/bin/pelorus-#{version}-macos-amd64.tar.gz"
      sha256 "X86_SHA256_PLACEHOLDER"
    end
  end

  def install
    bin.install "pelorus"
    generate_completions_from_executable(bin/"pelorus", "completions")
  end

  test do
    system "#{bin}/pelorus", "--version"
  end
end
