import ipaddress


def make_registrations():
    # ipaddress uses a homegrown internal netmask cache.
    # There aren't many netmasks - load all of them (to avoid nondeterminism):
    [ipaddress.IPv6Network._make_netmask(sz) for sz in range(129)]
    [ipaddress.IPv4Network._make_netmask(sz) for sz in range(33)]
