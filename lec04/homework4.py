
def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    month, day = date
    while True:
        month, day = next_month_day(month, day)
        if (month, day) in birthdays:
            return (month, day), birthdays[(month, day)]


def next_month_day(month, day):
    if month == 12 and day == 31:
        return 1, 1
    elif day == 30:
        return month + 1, 1
    else:
        return month, day + 1
    
