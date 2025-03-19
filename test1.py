from datetime import date

import logfire

logfire.configure()

# with logfire.span('Asking the user for their {question}', question='birthday'):  
#     user_input = input('When were you born [YYYY-mm-dd]? ')
#     dob = date.fromisoformat(user_input)  
#     age=(date.today() - dob)
#     logfire.debug('{dob=} {age=!r}', dob=dob, age=date.today() - dob)
#     logfire.info('The users age is {age}', age=age)
#     logfire.info('Another info logged.')

logfire.info("This is a test log from Logfire!", extra={"key": "value"})