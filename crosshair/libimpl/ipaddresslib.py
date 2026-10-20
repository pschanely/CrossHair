import ipaddress

from crosshair.core import register_patch, with_realized_args


def make_registrations():
    # ipaddress uses a homegrown internal netmask cache.
    # There aren't many netmasks - load all of them (to avoid nondeterminism):
    [ipaddress.IPv6Network._make_netmask(sz) for sz in range(129)]
    [ipaddress.IPv4Network._make_netmask(sz) for sz in range(33)]
    # These coerce their argument via __int__, which rejects a symbolic proxy;
    # realize the argument and defer to the real factory.
    register_patch(ipaddress.ip_network, with_realized_args(ipaddress.ip_network))
    register_patch(ipaddress.ip_interface, with_realized_args(ipaddress.ip_interface))
