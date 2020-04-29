---
layout: post
title:  "New Raspberry Pi 4"
date: 2020-04-28 16:18:00 -0400
categories: linux
---

I got a new Raspberry Pi 4! I wanted to install Arch on it, and here's how I set it up.

# Get started
The place to start is always the Arch wiki, and in this case there is an [installation page][] for Raspberry Pi 4 already. As long as you have a Linux box to run this on to get you started, you'll be fine. It only takes about 10 minutes.

At this point you should be able to login as `alarm`.

> If you find the excess console messages annoying, reduce the log level using `dmesg -n warn`.
> You can make it permanent by tacking on `loglevel=4` onto your kernel command line in `/boot/cmdline.txt`

## Wifi
I don't have Ethernet setup in my office so I couldn't connect it to Ethernet as the wiki suggested. Instead, I used a keyboard and a spare monitor to connect, and I used `wifi-menu`, part of the [netctl][] suite, to setup the wireless support from the command line. It uses a few text dialogs to generate and initiate a configuration for you in a few seconds. There are no driver issues but I wouldn't say it went "without a hitch".

```sh
# Login as root, or else use su
# The default root password is "root"

# Setup wifi. It will walk you through some dialogs.
# I assume you name your profile "home" but if you don't, then change "home" anywhere after this
wifi-menu
# Enable it at boot
netctl enable home
# You can probably avoid this reboot by restarting a few services,
# but I didn't find that set of services and rebooting a new rpi4 takes like 20 seconds anyway
reboot

# Log back in, doesn't need to be root
# the normal user is alarm and it's password is "alarm"

# You should be connected now!
ping google.com 
```

## Get a user and a hostname
As cool as it is always being root, I don't want everyone else to be.
```sh
# login as root
# Lock the normal user
passwd -l alarm
# Create your own (name it what you want)
useradd -m sean -s /bin/bash -U
passwd sean # Change your new account password (must do this the first time to log in)
passwd      # Change your root account password (else it will be "root")
hostnamectl set-hostname yabbadabbado
```

## Get a desktop
If you don't want a desktop, don't install one. This is my only Linux box, and I haven't tried out KDE in a while, so I got the 4GB model so that I could try out KDE.

Once again the Arch wiki does not disappoint. It explains in [KDE][] how to install what I'm looking for. But if you're into copy pasta, this is my suggestion:

* Update all the existing packages. No time like the present!
* Install [Xorg][] and the fbdev framebuffer driver you'll need for RPI4. (which I learned from trying to [install Arch on RPI3][]). Some suggest that you should use fbturbo but in my experience I didn't see a difference. Probably the C-compilers have already mastered the techniques used tweaking the driver for ARM, and there wasn't much more to get.
* [LightDM][] will give you a login screen, and I picked the typical GTK based greeter for it.
* Plasma will get you a window manager, and kde-applications will get you some everyday apps

```sh
# If you do these all at once you won't need to come back and back to check on it
pacman -Suy xorg xf86-video-fbdev lightdm lightdm-gtk-greeter plasma kde-applications
```

It will ask you a few questions, when there are multiple packages that would supply a feature you need. You are free to choose, but these were my selections:
* I use `libglvnd` to supply libgl. The others are for graphics cards I don't have.
* I use `libbluray` because I'm not installing Kodi.
* I use `phonon-qt5-vlc` because [the Phonon developers recommend VLC].
* I use `cronie` to supply cron, because it's the typical one.

Get a cup of coffee. It'll take at least 30 minutes. I picked an aluminum case that isn't great for my wifi and it took over an hour.
Once it's done, enable the login screen and reboot.

```sh
# Enable the login screen
systemctl enable lightdm

# This is another good time to reboot and check that all the drivers are working
reboot
```
Voila! You have a desktop!

## Graphics Drivers

You can test out the graphics using mesa's `glxinfo` and `glxgears` but don't expect to be blown away. The GPU isn't supported yet but while there is work underway to support it.
```sh
# as root
pacman -S mesa-demos
# not root, and in a graphical console (e.g. konsole, not ssh)
glxinfo | grep renderer # mentions llvmpipe, which does software rendering
# not root
glxgears # Gives pretty sad performance
```

Don't dispair! This is trivial to improve! As root, use `nano /boot/config.txt` to add this line at the beginning, then save and reboot. You'll be greeted by beautifully fast graphics and `glxinfo` will now mention V3D instead of llvmpipe. Nice, eh? 
```
# prepend this to /boot/config.txt as root and reboot
dtoverlay=vc4-fkms-v3d,cma-256
```
It works thanks to an open source driver for the VC4 GPU written by Eric Anholt, which you enable using a devicetree overlay named vc4-fkms-v3d. (There is another one with KMS instead of FKMS. FKMS is better supported but in practice neither have ever broken for me.) The CMA (contiguous memory allocator) option chooses how much memory you're willing to use exclusively for the GPU, and I chose the maximum (256MB) because I have enough for the CPU already, and this will improve graphics performance.

That's it! You're done!


[installation page]: https://archlinuxarm.org/platforms/armv8/broadcom/raspberry-pi-4
[install Arch on RPI3]: https://archlinuxarm.org/platforms/armv8/broadcom/raspberry-pi-3
[netctl]: https://wiki.archlinux.org/index.php/Netctl
[KDE]: https://wiki.archlinux.org/index.php/KDE
[Xorg]: https://wiki.archlinux.org/index.php/Xorg
[LightDM]: https://wiki.archlinux.org/index.php/LightDM
[the Phonon developers recommend VLC]: https://www.phoronix.com/scan.php?page=news_item&px=MTUwNDM