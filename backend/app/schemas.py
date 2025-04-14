from pydantic import BaseModel, EmailStr

# define how data is structured for input/output via the API
class SignupModel(BaseModel):
    fullname: str
    email: EmailStr
    password: str

class LoginModel(BaseModel):
    email: EmailStr
    password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str