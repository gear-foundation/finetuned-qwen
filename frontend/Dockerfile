FROM node:18 AS builder
WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install

COPY . .

ARG REACT_APP_API_URL 
ENV REACT_APP_API_URL=${REACT_APP_API_URL}
    
RUN yarn build

FROM nginx:alpine
ARG REACT_APP_API_URL 
ENV REACT_APP_API_URL=${REACT_APP_API_URL}
COPY --from=builder /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
