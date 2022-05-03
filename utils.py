import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def print_field(field, value, pfx):
    str = Gst.value_serialize(value)
    print("{0:s}  {1:15s}: {2:s}".format(
        pfx, GLib.quark_to_string(field), str))
    return True


def print_caps(caps, pfx):
    if not caps:
        return

    if caps.is_any():
        print("{0:s}ANY".format(pfx))
        return

    if caps.is_empty():
        print("{0:s}EMPTY".format(pfx))
        return

    for i in range(caps.get_size()):
        structure = caps.get_structure(i)
        print("{0:s}{1:s}".format(pfx, structure.get_name()))
        structure.foreach(print_field, pfx)

# prints information about a pad template (including its capabilities)


def print_pad_templates_information(factory):
    print("Pad templates for {0:s}".format(factory.get_name()))
    if factory.get_num_pad_templates() == 0:
        print("  none")
        return

    pads = factory.get_static_pad_templates()
    for pad in pads:
        padtemplate = pad.get()

        if pad.direction == Gst.PadDirection.SRC:
            print("  SRC template:", padtemplate.name_template)
        elif pad.direction == Gst.PadDirection.SINK:
            print("  SINK template:", padtemplate.name_template)
        else:
            print("  UNKNOWN template:", padtemplate.name_template)

        if padtemplate.presence == Gst.PadPresence.ALWAYS:
            print("    Availability: Always")
        elif padtemplate.presence == Gst.PadPresence.SOMETIMES:
            print("    Availability: Sometimes")
        elif padtemplate.presence == Gst.PadPresence.REQUEST:
            print("    Availability: On request")
        else:
            print("    Availability: UNKNOWN")

        if padtemplate.get_caps():
            print("    Capabilities:")
            print_caps(padtemplate.get_caps(), "      ")

        print("")