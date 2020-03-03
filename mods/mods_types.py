import arrow
import datetime
import json
import pytz

from json import JSONEncoder

def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default

def format_datetime(dt: datetime.datetime):
    s = ""
    if dt.year > 0:
        s += "%04d" % dt.year
    if dt.month > 0:
        s += "-%02d" % dt.month
    if dt.day > 0:
        s += "-%02d" % dt.day
    if dt.hour > 0:
        s += " %02d" % dt.hour
    if dt.minute > 0:
        s += ":%02d" % dt.minute
    if dt.second > 0:
        s += ":%02d" % dt.second
    if dt.microsecond > 0:
        s += ".%02d" % dt.microsecond
    return s


class TimeRange:
    delimiter = ','

    def __init__(
            self,
            beg: datetime.datetime,
            end: datetime.datetime,
            lclosed: bool,
            rclosed: bool
    ):
        self.beg = beg
        self.end = end
        self.lclosed = lclosed
        self.rclosed = rclosed

    @classmethod
    def from_str(cls, value: str):
        if isinstance(value, TimeRange):
            # TODO: inspect, why we need this hack
            return value
        value = value.strip()
        lclosed = value.startswith('<')
        rclosed = value.endswith('>')
        value = value[1:len(value) - 1]
        beg, end = value.split(TimeRange.delimiter, 1)
        # TODO: parse correct interval, if incomplete date is specified
        # 1) (2019,*; (2019-12,* should expand to the closest date after 2019-12-31 (consider time)
        # 3) *,2019); *,2019-01) should expand to the closest date before 2019-01-01
        # 2) <2019,*; <2019-01,*; <2019-01-01,* is OK
        beg = arrow.get(beg.strip()).datetime.replace(tzinfo=pytz.UTC)
        end = arrow.get(end.strip()).datetime.replace(tzinfo=pytz.UTC)
        return cls(beg, end, lclosed, rclosed)

    def is_lclosed(self):
        return self.lclosed

    def is_rclosed(self):
        return self.rclosed

    def to_str(self):
        return str("%s%s%s%s%s" % (
            '<' if self.lclosed else '(',
            format_datetime(self.beg),
            TimeRange.delimiter,
            format_datetime(self.end),
            '>' if self.rclosed else ')',
        ))

    def __repr__(self):
        return self.to_str()

    def __str__(self):
        return self.to_str()

    def toJSON(self):
        return self.to_str()

    def to_json(self):
        return self.to_str()
